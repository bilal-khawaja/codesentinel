"""Small account service API (FastAPI). Planted vulnerabilities and bugs; the labels are VULNERABLE_EXPECTED in code_analysis/evaluation.py."""
import hashlib
import os
import pickle
import re
import sqlite3

import requests
from fastapi import Depends, FastAPI
from pydantic import BaseModel

BASE_DIR = "/srv/reports"
DB_PATH = "users.db"

app = FastAPI()


class UserIn(BaseModel):
    username: str
    email: str
    password: str


def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def find_user(conn, username):
    cur = conn.cursor()
    cur.execute(f"SELECT id, email FROM users WHERE name = '{username}'")
    return cur.fetchone()


def ping_host(host):
    return os.system("ping -c 1 " + host)


def load_session(blob):
    return pickle.loads(blob)


def fetch_rates(url):
    response = requests.get(url, verify=False, timeout=5)
    return response.json()


def add_item(item, items=[]):
    items.append(item)
    return items


def total_price(prices):
    total = 0
    for i in range(len(prices) + 1):
        total += prices[i]
    return total


def extract_domain(email):
    match = re.match(r"[^@]+@(.+)", email)
    return match.group(1)


def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()


def read_report(name):
    with open(os.path.join(BASE_DIR, name)) as handle:
        return handle.read()


@app.post("/users")
def create_user(user: UserIn, conn=Depends(get_db)):
    conn.execute(
        "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
        (user.username, user.email, hash_password(user.password)),
    )
    conn.commit()
    return {"username": user.username}


@app.get("/users/{username}")
def get_user(username: str, conn=Depends(get_db)):
    row = find_user(conn, username)
    return {"id": row[0], "email": row[1]} if row else {}


@app.get("/users/{username}/domain")
def get_user_domain(username: str, conn=Depends(get_db)):
    row = find_user(conn, username)
    return {"domain": extract_domain(row[1]) if row else ""}


@app.get("/ping/{host}")
def ping(host: str):
    return {"exit_code": ping_host(host)}


@app.post("/session")
def open_session(blob: bytes):
    return load_session(blob)


@app.get("/rates")
def rates(url: str):
    return fetch_rates(url)


@app.post("/items")
def push_item(item: str):
    return add_item(item)


@app.post("/total")
def total(prices: list[float]):
    return {"total": total_price(prices)}


@app.get("/reports/{name}")
def report(name: str):
    return {"content": read_report(name)}
