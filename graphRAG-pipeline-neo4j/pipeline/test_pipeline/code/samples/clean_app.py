"""Small account service. Written to be correct; every pattern here is the safe version of one in vulnerable_app.py."""
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import subprocess
from pathlib import Path

import requests

BASE_DIR = Path("/srv/reports").resolve()
PAYMENT_API_KEY = os.environ.get("PAYMENT_API_KEY", "")


def find_user(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE name = ?", (username,))
    return cur.fetchone()


def ping_host(host):
    if not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host):
        raise ValueError("invalid host name")
    return subprocess.run(["ping", "-c", "1", host], capture_output=True, check=False).returncode


def load_session(blob):
    return json.loads(blob)


RATES_URL = "https://rates.example.com/v1/latest"


def fetch_rates(currency):
    response = requests.get(RATES_URL, params={"base": currency}, timeout=5)
    response.raise_for_status()
    return response.json()


def add_item(item, items=None):
    items = [] if items is None else list(items)
    items.append(item)
    return items


def total_price(prices):
    total = 0
    for price in prices:
        total += price
    return total


def extract_domain(email):
    match = re.match(r"[^@]+@(.+)", email)
    if match is None:
        return None
    return match.group(1)


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return salt.hex() + ":" + digest.hex()


def read_report(name):
    path = (BASE_DIR / name).resolve()
    if BASE_DIR not in path.parents:
        raise ValueError("path outside the report directory")
    with path.open() as handle:
        return handle.read()


def save_user(conn, username, email):
    try:
        conn.execute("INSERT INTO users (name, email) VALUES (?, ?)", (username, email))
        conn.commit()
    except sqlite3.Error:
        logging.exception("could not save user %s", username)
        raise
