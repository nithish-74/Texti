import unittest

from texti.secrets_store import KeyringSecretStore, MemorySecretStore


class FakeKeyring:
    """Mimics the keyring module, and enforces the Windows 2.5 KB limit."""

    LIMIT = 2560

    def __init__(self):
        self.d = {}

    def get_password(self, service, name):
        return self.d.get((service, name))

    def set_password(self, service, name, value):
        if len(value.encode("utf-8")) > self.LIMIT:
            raise ValueError("secret too large for this backend")
        self.d[(service, name)] = value

    def delete_password(self, service, name):
        del self.d[(service, name)]


class SecretStoreTests(unittest.TestCase):
    def test_memory_store_roundtrip(self):
        s = MemorySecretStore()
        self.assertIsNone(s.get("a"))
        s.set("a", "1")
        self.assertEqual(s.get("a"), "1")
        s.delete("a")
        self.assertIsNone(s.get("a"))

    def test_keyring_store_roundtrip_small(self):
        s = KeyringSecretStore(backend=FakeKeyring())
        s.set("k", "value")
        self.assertEqual(s.get("k"), "value")

    def test_large_value_is_split_under_the_backend_limit(self):
        kr = FakeKeyring()
        s = KeyringSecretStore(backend=kr)
        big = "t" * 9000  # an MSAL cache can be this large
        s.set("msal", big)
        self.assertEqual(s.get("msal"), big)
        self.assertGreater(len([k for k in kr.d if k[1].startswith("msal#")]), 3)

    def test_overwriting_with_shorter_value_leaves_no_stale_parts(self):
        kr = FakeKeyring()
        s = KeyringSecretStore(backend=kr)
        s.set("k", "a" * 5000)
        s.set("k", "short")
        self.assertEqual(s.get("k"), "short")
        self.assertEqual(sorted(k[1] for k in kr.d), ["k#0", "k#n"])

    def test_delete_removes_every_part(self):
        kr = FakeKeyring()
        s = KeyringSecretStore(backend=kr)
        s.set("k", "a" * 5000)
        s.delete("k")
        self.assertIsNone(s.get("k"))
        self.assertEqual(kr.d, {})

    def test_missing_part_reads_as_missing_not_corrupt_data(self):
        kr = FakeKeyring()
        s = KeyringSecretStore(backend=kr)
        s.set("k", "a" * 3000)
        del kr.d[("texti", "k#1")]
        self.assertIsNone(s.get("k"))


if __name__ == "__main__":
    unittest.main()
