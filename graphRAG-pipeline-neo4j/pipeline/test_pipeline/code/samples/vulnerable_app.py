"""Small account service. Planted vulnerabilities and bugs; see EXPECTED in test_analysis_accuracy.py."""
import hashlib
import os
import pickle
import re
import sqlite3

import requests

BASE_DIR = "/srv/reports"
PAYMENT_API_KEY = ""


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
