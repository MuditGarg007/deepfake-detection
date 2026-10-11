"""Clerk session verification.

The frontend signs users in with Clerk and sends the session token either as
`Authorization: Bearer <token>` (fetch/XHR calls) or in Clerk's `__session`
cookie (plain `<img>` loads, which cannot set headers). Both are short-lived
RS256 JWTs, checked here against the instance's public JWKS — no call to Clerk
per request.
"""

import base64
import logging

import jwt
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import config

logger = logging.getLogger(__name__)

LEEWAY_SECONDS = 10
PUBLIC_PATHS = {"/health", "/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"}


def frontend_api(publishable_key):
    # pk_test_<base64("clerk.example.com$")>
    encoded = publishable_key.split("_", 2)[-1]
    encoded += "=" * (-len(encoded) % 4)
    return base64.b64decode(encoded).decode().rstrip("$")


ISSUER = None
JWKS = None
if config.CLERK_PUBLISHABLE_KEY:
    ISSUER = f"https://{frontend_api(config.CLERK_PUBLISHABLE_KEY)}"
    JWKS = jwt.PyJWKClient(f"{ISSUER}/.well-known/jwks.json", cache_keys=True)
else:
    logger.warning("CLERK_PUBLISHABLE_KEY is not set — the API is open to anyone")


def session_token(request):
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.cookies.get("__session")


def current_user(request):
    """The Clerk user id set by RequireUser, or None when auth is disabled."""
    return getattr(request.state, "user_id", None)


class AuthError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def verify(token):
    """Return the Clerk user id the session token belongs to."""
    if not token:
        raise AuthError(401, "sign in to use the API")

    try:
        key = JWKS.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            issuer=ISSUER,
            leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWKClientError:
        logger.exception("could not fetch Clerk's signing keys")
        raise AuthError(503, "could not verify the session")
    except jwt.InvalidTokenError:
        raise AuthError(401, "your session has expired — sign in again")

    # `azp` is the origin the token was minted for. Rejecting foreign origins
    # stops a token issued to another site on the same Clerk instance.
    azp = claims.get("azp")
    if azp and azp not in config.CORS_ORIGINS:
        raise AuthError(401, "this session was issued for another site")

    return claims["sub"]


class RequireUser:
    """ASGI middleware that rejects unauthenticated API calls.

    A middleware rather than a route dependency so the token is checked before
    the request body is read: FastAPI resolves dependencies only after a
    multipart upload has fully arrived, and a slow upload can outlast Clerk's
    60-second session token.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            JWKS is None
            or scope["type"] != "http"
            or scope["method"] == "OPTIONS"
            or scope["path"] in PUBLIC_PATHS
        ):
            await self.app(scope, receive, send)
            return

        try:
            # verify() may fetch the JWKS over the network on a cache miss.
            user_id = await run_in_threadpool(verify, session_token(Request(scope)))
        except AuthError as exc:
            headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
            response = JSONResponse({"detail": exc.detail}, exc.status_code, headers)
            await response(scope, receive, send)
            return

        scope.setdefault("state", {})["user_id"] = user_id
        await self.app(scope, receive, send)
