"""A minimal Microsoft Graph client over an injectable transport.

Security details worth keeping:
- The access token is sent only to graph.microsoft.com. The pre-authenticated download
  URL Graph returns lives on a different host, so it is fetched WITHOUT the token. Python's
  default redirect handling would forward the Authorization header, so redirects are
  followed by hand.
- Retries only on throttling (429) and temporary errors (503), honouring Retry-After.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol
from urllib.parse import urlparse

GRAPH = "https://graph.microsoft.com/v1.0"


class GraphError(RuntimeError):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(f"Graph request failed with HTTP {status}" + (f": {message}" if message else ""))
        self.status = status


@dataclass
class Response:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


class Transport(Protocol):
    def get(self, url: str, headers: dict[str, str]) -> Response: ...
    def download(self, url: str, headers: dict[str, str], dest: Path) -> Response: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


class UrllibTransport:
    """Standard-library transport that never follows redirects on its own."""

    def __init__(self, timeout: float = 60.0) -> None:
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._timeout = timeout

    def _open(self, url: str, headers: dict[str, str]):
        req = urllib.request.Request(url, headers=headers)
        try:
            return self._opener.open(req, timeout=self._timeout)
        except urllib.error.HTTPError as e:  # includes 3xx because redirects are not followed
            return e

    def get(self, url: str, headers: dict[str, str]) -> Response:
        r = self._open(url, headers)
        try:
            return Response(r.status if hasattr(r, "status") else r.code, dict(r.headers.items()), r.read())
        finally:
            r.close()

    def download(self, url: str, headers: dict[str, str], dest: Path) -> Response:
        r = self._open(url, headers)
        try:
            status = r.status if hasattr(r, "status") else r.code
            if status == 200:
                with open(dest, "wb") as f:
                    while chunk := r.read(1024 * 1024):
                        f.write(chunk)
            return Response(status, dict(r.headers.items()))
        finally:
            r.close()


class GraphClient:
    def __init__(self, token: Callable[[], str], transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._token = token
        self._t = transport or UrllibTransport()
        self._sleep = sleep

    @staticmethod
    def _url(path_or_url: str) -> str:
        return path_or_url if path_or_url.startswith("https://") else GRAPH + path_or_url

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token()}", "Accept": "application/json"}

    def get_json(self, path_or_url: str) -> dict:
        url = self._url(path_or_url)
        if urlparse(url).hostname != "graph.microsoft.com":
            raise GraphError(0, "refusing to send credentials to a non-Graph host")
        for attempt in range(4):
            r = self._t.get(url, self._auth())
            if r.status in (429, 503) and attempt < 3:
                self._sleep(min(float(r.headers.get("Retry-After", 2 ** attempt)), 30.0))
                continue
            if r.status != 200:
                raise GraphError(r.status)
            return json.loads(r.body or b"{}")
        raise GraphError(429)

    def pages(self, path_or_url: str) -> Iterator[dict]:
        """Yield each page of a paged collection; the last page has the deltaLink, if any."""
        url: str | None = path_or_url
        while url:
            page = self.get_json(url)
            yield page
            url = page.get("@odata.nextLink")

    def download(self, item_id: str, download_url: str | None, dest: Path) -> None:
        """Save a file's bytes to ``dest``. Never sends the Microsoft token to a non-Graph host."""
        if download_url:
            r = self._t.download(download_url, {}, dest)  # pre-authenticated: no token
            if r.status != 200:
                raise GraphError(r.status)
            return
        url = f"{GRAPH}/me/drive/items/{item_id}/content"
        headers = self._auth()
        for _ in range(5):
            r = self._t.download(url, headers, dest)
            if r.status == 200:
                return
            if r.status in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                url = r.headers["Location"]
                if urlparse(url).hostname != "graph.microsoft.com":
                    headers = {}  # drop the token once we leave Graph
                continue
            raise GraphError(r.status)
        raise GraphError(310, "too many redirects")
