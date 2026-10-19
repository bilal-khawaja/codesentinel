"""Small account service API (FastAPI). Written to be correct; every pattern here is the safe version of one in vulnerable_app.py."""
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional

import requests
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path("/srv/reports").resolve()
DB_PATH = os.environ.get("DB_PATH", "users.db")
PAYMENT_API_KEY = os.environ.get("PAYMENT_API_KEY", "")
RATES_URL = "https://rates.example.com/v1/latest"
HOST_PATTERN = re.compile(r"[A-Za-z0-9.-]{1,253}")
CURRENCY_PATTERN = re.compile(r"[A-Z]{3}")
EMAIL_PATTERN = re.compile(r"[^@\s]+@([^@\s]+)")

app = FastAPI()


class UserIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


def get_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def find_user(conn: sqlite3.Connection, username: str) -> Optional[tuple]:
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE name = ?", (username,))
    return cur.fetchone()


def ping_host(host: str) -> int:
    if not HOST_PATTERN.fullmatch(host):
        raise ValueError("invalid host name")
    return subprocess.run(["ping", "-c", "1", host], capture_output=True, check=False, timeout=10).returncode


def load_session(blob: str) -> dict:
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("session must be a JSON object")
    return data


def fetch_rates(currency: str) -> dict:
    if not CURRENCY_PATTERN.fullmatch(currency):
        raise ValueError("invalid currency code")
    headers = {"Authorization": f"Bearer {PAYMENT_API_KEY}"} if PAYMENT_API_KEY else {}
    response = requests.get(RATES_URL, params={"base": currency}, headers=headers, timeout=5)
    response.raise_for_status()
    return response.json()


def add_item(item: str, items: Optional[List[str]] = None) -> List[str]:
    items = [] if items is None else list(items)
    items.append(item)
    return items


def total_price(prices: List[float]) -> float:
    total = 0.0
    for price in prices:
        total += price
    return total


def extract_domain(email: str) -> Optional[str]:
    match = EMAIL_PATTERN.fullmatch(email)
    if match is None:
        return None
    return match.group(1)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return salt.hex() + ":" + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    salt_hex, _, digest_hex = stored.partition(":")
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return hmac.compare_digest(actual, expected)


def read_report(name: str) -> str:
    path = (BASE_DIR / name).resolve()
    if not path.is_relative_to(BASE_DIR):
        raise ValueError("path outside the report directory")
    if not path.is_file():
        raise FileNotFoundError(name)
    with path.open() as handle:
        return handle.read()


def save_user(conn: sqlite3.Connection, username: str, email: str, password_hash: str) -> None:
    try:
        conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        logging.exception("could not save user %s", username)
        raise


@app.post("/users", status_code=201)
def create_user(user: UserIn, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    if extract_domain(user.email) is None:
        raise HTTPException(status_code=422, detail="invalid email address")
    try:
        save_user(conn, user.username, user.email, hash_password(user.password))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="user already exists")
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="could not save user")
    return {"username": user.username}


@app.get("/users/{username}")
def get_user(username: str, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    row = find_user(conn, username)
    if row is None:
        raise HTTPException(status_code=404, detail="user not found")
    return {"id": row[0], "email": row[1]}


@app.get("/users/{username}/domain")
def get_user_domain(username: str, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    row = find_user(conn, username)
    if row is None:
        raise HTTPException(status_code=404, detail="user not found")
    domain = extract_domain(row[1])
    if domain is None:
        raise HTTPException(status_code=422, detail="stored email is invalid")
    return {"domain": domain}


@app.get("/ping/{host}")
def ping(host: str) -> dict:
    try:
        return {"exit_code": ping_host(host)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ping timed out")


@app.post("/session")
def open_session(blob: str) -> dict:
    try:
        return load_session(blob)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/rates/{currency}")
def rates(currency: str) -> dict:
    try:
        return fetch_rates(currency)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="rates service unavailable")


@app.post("/items")
def push_item(item: str) -> List[str]:
    return add_item(item)


@app.post("/total")
def total(prices: List[float]) -> dict:
    return {"total": total_price(prices)}


@app.get("/reports/{name}")
def report(name: str) -> dict:
    try:
        return {"content": read_report(name)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="report not found")
