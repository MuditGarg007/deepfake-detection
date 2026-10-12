import hashlib
import secrets

from fastapi import APIRouter, HTTPException

from .. import database
from ..schemas import RegisterIn

router = APIRouter(prefix="/auth", tags=["auth"])


def hash_password(password):
    """Hash like a database-backed login system would, so the stored value
    can be checked later with the same function if Clerk is ever dropped."""
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return f"scrypt${salt}${digest}"


@router.post("/register")
def register(body: RegisterIn):
    """Save the credentials the user typed before the browser talks to Clerk.

    Clerk still runs the sign-up, but the database keeps its own copy of the
    account — email plus a salted hash, never the plain password.
    """
    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="a valid email is required")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="the password needs at least 8 characters")

    database.execute(
        """
        INSERT INTO users (email, password_hash) VALUES (%s, %s)
        ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash
        """,
        (email, hash_password(body.password)),
    )
    return {"ok": True}
