import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .document import Document

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}

_SSRF_MAX_REDIRECTS = 5


def is_public_ip(ip_str: str) -> bool:
    """Returns False for private, loopback, link-local, multicast, reserved or unspecified IPs."""
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def assert_public_http_url(url: str):
    """Guards against SSRF: only http(s) URLs resolving to public IPs are allowed."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("Only http(s) URLs are allowed")
    if not p.hostname:
        raise ValueError("URL is missing a hostname")
    try:
        addrinfo = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve host: {p.hostname}") from exc
    resolved_ips = {sockaddr[0] for _, _, _, _, sockaddr in addrinfo}
    if not resolved_ips or not all(is_public_ip(ip) for ip in resolved_ips):
        raise ValueError(f"Access to internal addresses is blocked: {p.hostname}")


def fetch_public_url(url: str, session: requests.Session = None, max_redirects: int = _SSRF_MAX_REDIRECTS, **kwargs):
    """Safely GETs a URL, validating every redirect hop resolves only to public IPs."""
    session = session or requests
    current_url = url
    for _ in range(max_redirects + 1):
        assert_public_http_url(current_url)
        response = session.get(current_url, allow_redirects=False, **kwargs)
        if response.is_redirect or response.is_permanent_redirect:
            next_url = response.headers.get('Location')
            if not next_url:
                return response, current_url
            current_url = requests.compat.urljoin(current_url, next_url)
            continue
        return response, current_url
    raise ValueError("Too many redirects")


def get_documents_from_web_page(source_url: str):
    if not source_url.startswith(('http://', 'https://')):
        source_url = 'https://' + source_url

    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)

    response, final_url = fetch_public_url(source_url, session=session, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return [Document(page_content=text, metadata={"source": final_url})]
