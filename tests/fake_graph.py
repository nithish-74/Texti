"""An in-memory fake of the parts of Microsoft Graph that texti uses."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from texti.onedrive.graph import Response

DOWNLOAD_HOST = "download.example.net"


class FakeDrive:
    """Implements the Transport protocol. Items: {id: {...Graph item...}} plus file bytes."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.content: dict[str, bytes] = {}
        self.version = 0
        self.log: list[tuple[str, str, dict]] = []  # (kind, url, headers) for every request
        self.deleted_since: dict[int, list[str]] = {}
        self.changed_at: dict[str, int] = {}
        self.delta_status = 200  # set to 400 to simulate a drive that does not support folder delta
        self.throttle_next = 0  # number of 429 responses to return before succeeding
        self.use_redirect = False  # serve /content as a 302 to the download host
        self.add("root", "", folder=True, parent_path=None)

    # -- building the drive -------------------------------------------------------------

    def add(self, item_id, name, folder=False, parent_path="/drive/root:", data: bytes | None = None, hash_=None):
        self.version += 1
        item = {"id": item_id, "name": name, "eTag": f"e{self.version}", "cTag": f"c{self.version}",
                "parentReference": {"path": parent_path} if parent_path is not None else {}}
        if folder:
            item["folder"] = {"childCount": 0}
        else:
            item["file"] = {"hashes": {"sha1Hash": hash_ or f"sha1-{self.version}"}}
            item["size"] = len(data or b"")
            item["@microsoft.graph.downloadUrl"] = f"https://{DOWNLOAD_HOST}/{item_id}"
            self.content[item_id] = data or b""
        self.items[item_id] = item
        self.changed_at[item_id] = self.version

    def update(self, item_id, data: bytes, hash_: str):
        self.version += 1
        self.content[item_id] = data
        self.items[item_id]["file"]["hashes"]["sha1Hash"] = hash_
        self.items[item_id]["size"] = len(data)
        self.changed_at[item_id] = self.version

    def remove(self, item_id):
        self.version += 1
        del self.items[item_id]
        self.content.pop(item_id, None)
        self.deleted_since.setdefault(self.version, []).append(item_id)

    # -- Transport protocol --------------------------------------------------------------

    def _ok(self, obj) -> Response:
        return Response(200, {}, json.dumps(obj).encode())

    def get(self, url: str, headers: dict) -> Response:
        self.log.append(("get", url, dict(headers)))
        u = urlparse(url)
        assert u.hostname == "graph.microsoft.com", f"token sent to {u.hostname}"
        if self.throttle_next > 0:
            self.throttle_next -= 1
            return Response(429, {"Retry-After": "1"}, b"")
        parts = [p for p in u.path.split("/") if p]  # v1.0 me drive items {id} [children|delta]
        if "delta" in parts:
            return self._delta(parts, parse_qs(u.query))
        if parts[-1] == "children":
            parent = parts[-2] if parts[-2] != "root" else "root"
            kids = [self._with_name_path(i) for i in self.items.values() if self._parent_id(i) == parent]
            return self._ok({"value": kids})
        item_id = parts[-1]
        if item_id not in self.items:
            return Response(404)
        return self._ok(self.items[item_id])

    @staticmethod
    def full_path(item) -> str:
        return item["parentReference"].get("path", "") + "/" + item["name"]

    def _parent_id(self, item):
        path = item["parentReference"].get("path", "")
        if path == "/drive/root:":
            return "root"
        for other in self.items.values():
            if "folder" in other and other["id"] != "root" and self.full_path(other) == path:
                return other["id"]
        return None

    def _with_name_path(self, item):
        return {k: v for k, v in item.items() if k != "parentReference"}

    def _delta(self, parts, query) -> Response:
        if self.delta_status != 200:
            return Response(self.delta_status)
        since = int(query.get("token", ["0"])[0])
        folder_id = parts[parts.index("items") + 1]
        folder = self.items[folder_id]
        inside = self.full_path(folder)
        stubs = []
        for i in self.items.values():
            if self.changed_at.get(i["id"], 0) > since and (
                i["parentReference"].get("path", "").startswith(inside) or i["id"] == folder_id
            ):
                stub = {k: v for k, v in i.items() if k in ("id", "name", "file", "folder", "size")}
                stubs.append(stub)  # real delta responses omit the path: so does the fake
        for ver, ids in self.deleted_since.items():
            if ver > since:
                stubs.extend({"id": d, "deleted": {"state": "deleted"}} for d in ids)
        return self._ok({"value": stubs,
                         "@odata.deltaLink": f"https://graph.microsoft.com/v1.0/me/drive/items/{folder_id}/delta?token={self.version}"})

    def download(self, url: str, headers: dict, dest: Path) -> Response:
        self.log.append(("download", url, dict(headers)))
        u = urlparse(url)
        if u.hostname == DOWNLOAD_HOST:
            assert "Authorization" not in headers, "Microsoft token was sent to the download host"
            item_id = u.path.strip("/")
            Path(dest).write_bytes(self.content[item_id])
            return Response(200)
        # Graph /content endpoint
        item_id = u.path.split("/")[-2]
        if self.use_redirect:
            return Response(302, {"Location": f"https://{DOWNLOAD_HOST}/{item_id}"})
        Path(dest).write_bytes(self.content[item_id])
        return Response(200)

    # -- helpers for assertions ------------------------------------------------------------

    def downloads(self) -> int:
        return sum(1 for kind, _, _ in self.log if kind == "download")
