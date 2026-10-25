"""Microsoft sign-in with MSAL (interactive PKCE), tokens kept in the OS keychain.

Scope is read-only. MSAL adds ``offline_access`` (the refresh token) by itself and
rejects it if passed explicitly, so only ``Files.Read`` is listed here.

You need an app registration in Microsoft Entra: a public client with the redirect URI
``http://localhost`` and the delegated permission ``Files.Read``. Its client ID goes in
the TEXTI_CLIENT_ID environment variable.
"""

from __future__ import annotations

import os

from ..secrets_store import SecretStore

SCOPES = ["Files.Read"]
CACHE_KEY = "msal-token-cache"
DEFAULT_AUTHORITY = "https://login.microsoftonline.com/common"  # personal and work/school accounts


class AuthError(RuntimeError):
    pass


def client_id() -> str:
    cid = os.environ.get("TEXTI_CLIENT_ID", "").strip()
    if not cid:
        raise AuthError(
            "TEXTI_CLIENT_ID is not set. Register a public-client app in Microsoft Entra "
            "(redirect URI http://localhost, delegated permission Files.Read) and set its client ID."
        )
    return cid


class Authenticator:
    def __init__(self, secrets: SecretStore, cid: str | None = None, authority: str | None = None, msal_module=None):
        if msal_module is None:
            try:
                import msal as msal_module  # type: ignore[no-redef]
            except ImportError as e:
                raise AuthError("msal is not installed: pip install msal") from e
        self._msal = msal_module
        self._secrets = secrets
        self._cache = msal_module.SerializableTokenCache()
        saved = secrets.get(CACHE_KEY)
        if saved:
            self._cache.deserialize(saved)
        self._app = msal_module.PublicClientApplication(
            cid or client_id(),
            authority=authority or os.environ.get("TEXTI_AUTHORITY", DEFAULT_AUTHORITY),
            token_cache=self._cache,
        )

    def _persist(self) -> None:
        if self._cache.has_state_changed:
            self._secrets.set(CACHE_KEY, self._cache.serialize())

    def login(self) -> str:
        """Open the browser for sign-in; returns the signed-in account name."""
        result = self._app.acquire_token_interactive(scopes=SCOPES)
        self._persist()
        if "access_token" not in result:
            raise AuthError(f"sign-in failed: {result.get('error', 'unknown error')}")
        return (result.get("id_token_claims") or {}).get("preferred_username", "signed in")

    def token(self) -> str:
        """A valid access token, refreshed silently. Raises if the user has not logged in."""
        accounts = self._app.get_accounts()
        if not accounts:
            raise AuthError("not signed in: run `texti login`")
        result = self._app.acquire_token_silent(SCOPES, account=accounts[0])
        self._persist()
        if not result or "access_token" not in result:
            raise AuthError("sign-in expired: run `texti login` again")
        return result["access_token"]

    def logout(self) -> None:
        for account in self._app.get_accounts():
            self._app.remove_account(account)
        self._secrets.delete(CACHE_KEY)
