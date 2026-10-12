from secrets import compare_digest

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from .. import config, database
from ..auth import ADMIN_COOKIE, admin_user, make_admin_token
from ..schemas import AdminLoginIn, AdminOut

router = APIRouter(prefix="/admin", tags=["admin"])

TABLE_ROW_LIMIT = 500


def require_admin(request: Request):
    # RequireUser lets Clerk users through too, so admin-only routes check
    # the admin cookie themselves.
    if admin_user(request) is None:
        raise HTTPException(status_code=401, detail="not signed in as admin")


@router.post("/login", response_model=AdminOut)
def login(body: AdminLoginIn, request: Request, response: Response):
    if not config.ADMIN_USERNAME or not config.ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="admin login is not set up")

    username_ok = compare_digest(body.username.encode(), config.ADMIN_USERNAME.encode())
    password_ok = compare_digest(body.password.encode(), config.ADMIN_PASSWORD.encode())
    if not (username_ok and password_ok):
        raise HTTPException(status_code=401, detail="wrong admin username or password")

    response.set_cookie(
        ADMIN_COOKIE,
        make_admin_token(),
        max_age=config.ADMIN_SESSION_SECONDS,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )
    return AdminOut(username=config.ADMIN_USERNAME)


@router.post("/logout", status_code=204)
def logout(response: Response):
    response.delete_cookie(ADMIN_COOKIE)


@router.get("/me", response_model=AdminOut, dependencies=[Depends(require_admin)])
def me():
    return AdminOut(username=config.ADMIN_USERNAME)


@router.get("/tables", dependencies=[Depends(require_admin)])
def tables():
    return database.table_names()


@router.get("/tables/{name}", dependencies=[Depends(require_admin)])
def table(name: str):
    if name not in database.table_names():
        raise HTTPException(status_code=404, detail="no such table")
    columns, rows = database.fetch_table(name, TABLE_ROW_LIMIT)
    return {"name": name, "columns": columns, "rows": rows, "limit": TABLE_ROW_LIMIT}
