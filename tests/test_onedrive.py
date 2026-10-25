import tempfile
import unittest
from pathlib import Path

from texti.config import FolderConfig
from texti.index.store import Store
from texti.onedrive.auth import AuthError, Authenticator
from texti.onedrive.graph import GraphClient, GraphError, Response
from texti.onedrive.sync import content_hash, item_path, list_folders, sync_folder
from texti.retrieve.search import search
from texti.secrets_store import MemorySecretStore
from tests import fixtures as fx
from tests.fake_graph import DOWNLOAD_HOST, FakeDrive
from tests.helpers import Env, text_image

SLIDE = ["Lecture 3: Binary Search Trees", "A binary search tree keeps smaller keys on the left."]
ID_SCAN = ["Government ID", f"Aadhaar number {fx.AADHAAR_SPACED}"]


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.drive = FakeDrive()
        self.drive.add("F1", "College", folder=True)
        self.drive.add("F2", "Maths", folder=True, parent_path="/drive/root:/College")
        self.drive.add("a", "slide.png", parent_path="/drive/root:/College", data=text_image(SLIDE))
        self.drive.add("b", "deep.png", parent_path="/drive/root:/College/Maths", data=text_image(SLIDE[:1] + ["Definition of a node."]))
        self.drive.add("c", "notes.txt", parent_path="/drive/root:/College", data=b"ignored")
        self.client = GraphClient(lambda: "TOKEN", self.drive, sleep=lambda s: None)
        self.folder = FolderConfig(id="F1", name="College", language="eng", tier=1)
        # Redirect the system temp dir so leaked temp files are detectable.
        self.tmp_root = Path(tempfile.mkdtemp(prefix="m2t-tmproot-"))
        self._old_tmp = tempfile.tempdir
        tempfile.tempdir = str(self.tmp_root)

    def tearDown(self):
        tempfile.tempdir = self._old_tmp
        self.env.close()

    def sync(self, process=None):
        kw = {"process": process} if process else {}
        return sync_folder(self.client, self.env.store, self.env.embedder, self.env.cfg, self.folder, **kw)

    def leftovers(self):
        return list(self.tmp_root.iterdir())

    # -- first sync -----------------------------------------------------------------------

    def test_first_sync_processes_supported_files_only(self):
        s = self.sync()
        # Skipped: the picked folder itself (delta returns it), the Maths sub-folder, and notes.txt.
        self.assertEqual((s.processed, s.skipped, s.failed), (2, 3, 0))
        res = search(self.env.store, self.env.embedder, "binary search tree", record_usage=False)
        self.assertTrue(res.hits)

    def test_the_picked_folders_tier_rule_applies_even_if_not_in_saved_config(self):
        self.sync()
        # Both files are in a Tier 1 folder; neither may fall back to the default Tier 2.
        self.assertEqual(self.env.store.stats()["files_by_tier"], {1: 2, 2: 0, 3: 0})

    def test_paths_are_relative_to_the_picked_folder(self):
        self.sync()
        res = search(self.env.store, self.env.embedder, "definition of a node", record_usage=False)
        self.assertIn("College/Maths/deep.png", {h.source_path for h in res.hits})

    def test_folder_rule_and_content_raise_to_tier_3(self):
        self.drive.add("id1", "scan.png", parent_path="/drive/root:/College", data=text_image(ID_SCAN))
        self.sync()
        stats = self.env.store.stats()
        self.assertGreater(stats["chunks_by_tier"][3], 0)
        self.assertGreater(stats["chunks_by_tier"][1], 0)

    # -- incremental sync -------------------------------------------------------------------

    def test_second_sync_does_nothing(self):
        self.sync()
        before = self.drive.downloads()
        s = self.sync()
        self.assertEqual((s.processed, s.deleted, s.failed), (0, 0, 0))
        self.assertEqual(self.drive.downloads(), before)

    def test_changed_file_is_reprocessed_and_deleted_file_is_removed(self):
        self.sync()
        self.drive.update("a", text_image(["Lecture 9: Graph Algorithms", "Dijkstra finds shortest paths."]), "sha1-new")
        self.drive.remove("b")
        s = self.sync()
        self.assertEqual((s.processed, s.deleted), (1, 1))
        res = search(self.env.store, self.env.embedder, "dijkstra shortest paths", record_usage=False)
        self.assertIn("dijkstra", " ".join(h.text for h in res.hits).lower())
        gone = search(self.env.store, self.env.embedder, "definition of a node", record_usage=False)
        self.assertNotIn("College/Maths/deep.png", {h.source_path for h in gone.hits})

    def test_delta_failure_falls_back_to_full_scan_and_skips_by_hash(self):
        self.drive.delta_status = 400
        s1 = self.sync()
        self.assertTrue(s1.full_scan)
        self.assertEqual(s1.processed, 2)
        before = self.drive.downloads()
        s2 = self.sync()
        self.assertTrue(s2.full_scan)
        self.assertEqual((s2.processed, s2.unchanged), (0, 2))
        self.assertEqual(self.drive.downloads(), before, "unchanged files must not be downloaded again")

    # -- failures and cleanup -----------------------------------------------------------------

    def test_temp_files_are_deleted_after_success(self):
        self.sync()
        self.assertEqual(self.leftovers(), [])

    def test_temp_files_are_deleted_after_a_failure(self):
        def boom(*a, **kw):
            raise RuntimeError("processing exploded")

        s = self.sync(process=boom)
        self.assertEqual(s.failed, 2)
        self.assertEqual(self.leftovers(), [], "temp download must be removed even when processing fails")

    def test_error_messages_are_not_recorded(self):
        def boom(*a, **kw):
            raise RuntimeError("secret path College/Maths/deep.png")

        s = self.sync(process=boom)
        self.assertEqual(set(s.errors), {"RuntimeError"})

    def test_failed_files_are_retried_because_the_delta_link_is_not_saved(self):
        def boom(*a, **kw):
            raise RuntimeError("x")

        self.sync(process=boom)
        self.assertIsNone(self.env.store.get_delta_link("F1"))
        s = self.sync()
        self.assertEqual((s.processed, s.failed), (2, 0))
        self.assertIsNotNone(self.env.store.get_delta_link("F1"))

    # -- token handling -------------------------------------------------------------------------

    def test_token_is_never_sent_to_the_download_host(self):
        self.sync()
        for kind, url, headers in self.drive.log:
            if DOWNLOAD_HOST in url:
                self.assertNotIn("Authorization", headers)
            else:
                self.assertEqual(headers.get("Authorization"), "Bearer TOKEN")

    def test_redirected_content_download_drops_the_token(self):
        self.drive.use_redirect = True
        for item in self.drive.items.values():
            item.pop("@microsoft.graph.downloadUrl", None)  # force the /content endpoint
        self.sync()
        downloads = [(url, h) for kind, url, h in self.drive.log if kind == "download"]
        self.assertTrue(any(DOWNLOAD_HOST in url for url, _ in downloads))
        for url, headers in downloads:
            if DOWNLOAD_HOST in url:
                self.assertNotIn("Authorization", headers)

    def test_graph_client_refuses_to_send_the_token_elsewhere(self):
        with self.assertRaises(GraphError):
            self.client.get_json("https://evil.example.com/steal")
        self.assertEqual([l for l in self.drive.log if "evil" in l[1]], [])

    def test_throttling_is_retried(self):
        self.drive.throttle_next = 2
        self.assertEqual(self.client.get_json("/me/drive/items/F1")["name"], "College")

    def test_persistent_throttling_eventually_fails(self):
        self.drive.throttle_next = 99
        with self.assertRaises(GraphError):
            self.client.get_json("/me/drive/items/F1")

    # -- helpers ----------------------------------------------------------------------------------

    def test_list_folders(self):
        names = [f["name"] for f in list_folders(self.client, "root")]
        self.assertEqual(names, ["College"])
        self.assertEqual([f["name"] for f in list_folders(self.client, "F1")], ["Maths"])

    def test_item_path_and_hash_helpers(self):
        item = {"name": "x y.png", "parentReference": {"path": "/drive/root:/My%20Folder/Sub"},
                "file": {"hashes": {"quickXorHash": "abc"}}}
        self.assertEqual(item_path(item), "My Folder/Sub/x y.png")
        self.assertEqual(content_hash(item), "quickXorHash:abc")
        self.assertEqual(content_hash({"cTag": "t"}), "tag:t")
        self.assertIsNone(content_hash({}))
        self.assertEqual(item_path({"name": "top.png", "parentReference": {"path": "/drive/root:"}}), "top.png")


class FakeCache:
    has_state_changed = False

    def __init__(self):
        self.data = None

    def deserialize(self, s):
        self.data = s

    def serialize(self):
        self.has_state_changed = False
        return self.data or "{}"


class FakeApp:
    def __init__(self, cid, authority=None, token_cache=None):
        self.cache, self.accounts = token_cache, []
        FakeApp.last = self
        self.scopes_seen = []

    def acquire_token_interactive(self, scopes):
        self.scopes_seen.append(scopes)
        self.accounts = [{"username": "u@example.com"}]
        self.cache.data, self.cache.has_state_changed = '{"RefreshToken": "x"}', True
        return {"access_token": "AT", "id_token_claims": {"preferred_username": "u@example.com"}}

    def get_accounts(self):
        return list(self.accounts)

    def acquire_token_silent(self, scopes, account):
        self.scopes_seen.append(scopes)
        return {"access_token": "AT2"}

    def remove_account(self, a):
        self.accounts.remove(a)


class FakeMsal:
    SerializableTokenCache = FakeCache
    PublicClientApplication = FakeApp


class AuthTests(unittest.TestCase):
    def test_login_saves_cache_to_secret_store_and_token_refreshes_silently(self):
        secrets = MemorySecretStore()
        auth = Authenticator(secrets, cid="cid", msal_module=FakeMsal)
        self.assertEqual(auth.login(), "u@example.com")
        self.assertIn("RefreshToken", secrets.get("msal-token-cache"))
        again = Authenticator(secrets, cid="cid", msal_module=FakeMsal)
        self.assertEqual(again._cache.data, secrets.get("msal-token-cache"))

    def test_scope_is_read_only_and_does_not_list_offline_access(self):
        auth = Authenticator(MemorySecretStore(), cid="cid", msal_module=FakeMsal)
        auth.login()
        auth.token()
        for scopes in FakeApp.last.scopes_seen:
            self.assertEqual(scopes, ["Files.Read"])  # MSAL adds offline_access itself

    def test_token_without_login_gives_a_clear_error(self):
        auth = Authenticator(MemorySecretStore(), cid="cid", msal_module=FakeMsal)
        with self.assertRaises(AuthError) as cm:
            auth.token()
        self.assertIn("login", str(cm.exception))

    def test_logout_clears_the_stored_cache(self):
        secrets = MemorySecretStore()
        auth = Authenticator(secrets, cid="cid", msal_module=FakeMsal)
        auth.login()
        auth.logout()
        self.assertIsNone(secrets.get("msal-token-cache"))
        with self.assertRaises(AuthError):
            auth.token()

    def test_missing_client_id_explains_how_to_register(self):
        import os

        old = os.environ.pop("TEXTI_CLIENT_ID", None)
        try:
            with self.assertRaises(AuthError) as cm:
                Authenticator(MemorySecretStore(), msal_module=FakeMsal)
            self.assertIn("TEXTI_CLIENT_ID", str(cm.exception))
        finally:
            if old is not None:
                os.environ["TEXTI_CLIENT_ID"] = old


if __name__ == "__main__":
    unittest.main()
