"""Supabase token verification for routes that handle private financial data."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    user_id: str | None


def require_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> CurrentUser:
    """Verify a Supabase access token and return its immutable user id.

    Local development without Supabase keeps the legacy unauthenticated path
    available; once ``SUPABASE_URL`` is configured, every private route
    requires a valid bearer token.
    """
    if not config.SUPABASE_URL:
        return CurrentUser(user_id=None)
    if not config.SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Supabase authentication is misconfigured.")
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in is required.")
    try:
        from supabase import create_client

        result = create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY).auth.get_user(credentials.credentials)
        if result.user is None:
            raise ValueError("No authenticated user")
        return CurrentUser(user_id=result.user.id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- auth provider errors are intentionally not exposed
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Your session is invalid or expired.") from exc
