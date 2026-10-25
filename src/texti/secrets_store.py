"""Secret storage: OS keychain in production, in-memory for tests.

Windows Credential Manager limits one secret to about 2.5 KB, and an MSAL token cache
(refresh + access + id tokens) can exceed that. KeyringSecretStore therefore splits
values across several keychain entries.
"""

from __future__ import annotations

from typing import Protocol

SERVICE = "texti"
_CHUNK = 1000  # characters per keychain entry, safely under the Windows limit


class SecretStore(Protocol):
    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class MemorySecretStore:
    """Non-persistent store for tests."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._data.get(name)

    def set(self, name: str, value: str) -> None:
        self._data[name] = value

    def delete(self, name: str) -> None:
        self._data.pop(name, None)


class KeyringSecretStore:
    """OS keychain through the ``keyring`` library, with transparent chunking."""

    def __init__(self, service: str = SERVICE, backend=None) -> None:
        self._service = service
        self._kr = backend  # injectable for tests; resolved lazily otherwise

    def _keyring(self):
        if self._kr is None:
            import keyring  # imported lazily so the rest of the package works without it

            self._kr = keyring
        return self._kr

    def get(self, name: str) -> str | None:
        kr = self._keyring()
        count = kr.get_password(self._service, f"{name}#n")
        if count is None:
            return None
        parts = []
        for i in range(int(count)):
            part = kr.get_password(self._service, f"{name}#{i}")
            if part is None:  # corrupt or partially deleted entry
                return None
            parts.append(part)
        return "".join(parts)

    def set(self, name: str, value: str) -> None:
        kr = self._keyring()
        self.delete(name)  # drop stale chunks from a previous, longer value
        pieces = [value[i : i + _CHUNK] for i in range(0, len(value), _CHUNK)] or [""]
        for i, piece in enumerate(pieces):
            kr.set_password(self._service, f"{name}#{i}", piece)
        kr.set_password(self._service, f"{name}#n", str(len(pieces)))

    def delete(self, name: str) -> None:
        kr = self._keyring()
        count = kr.get_password(self._service, f"{name}#n")
        if count is None:
            return
        for i in range(int(count)):
            try:
                kr.delete_password(self._service, f"{name}#{i}")
            except Exception:  # noqa: BLE001 - entry already gone
                pass
        try:
            kr.delete_password(self._service, f"{name}#n")
        except Exception:  # noqa: BLE001
            pass
