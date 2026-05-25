import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


_public_servers = [
    "https://www.google.com/",
    "https://www.cloudflare.com/",
    "https://www.opendns.com/"
]


def check_internet_connection(timeout: float = 5.0) -> bool:
    """Checks the internet connection by sending requests to public servers."""

    def probe(url: str) -> bool:
        """Sends a request and returns whether server responded or not."""

        try:
            requests.head(url=url, timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            logging.info(f'Failed to connect to {url}')
            return False

    with ThreadPoolExecutor(max_workers=len(_public_servers)) as executor:
        futures = [executor.submit(probe, url) for url in _public_servers]

        for fut in as_completed(futures):
            if fut.result():
                return True
    return False
