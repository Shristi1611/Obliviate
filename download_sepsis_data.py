"""
download_sepsis_data.py — Recursively downloads the PhysioNet 2019 Sepsis
Challenge training data, without needing wget/curl at all.

Run this ONCE on your own machine (not needed again after this):
    pip install requests beautifulsoup4
    python download_sepsis_data.py

It saves files into ./sepsis_raw/ -- matching where dataset.py expects to
find them (RAW_DATA_DIR).
"""
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"
OUT_DIR = os.path.join(os.path.dirname(__file__), "sepsis_raw", "training")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ObliviateProjectDownloader/1.0)"}

# A persistent session reuses one underlying TCP connection across requests
# instead of opening a brand new connection every single file -- this is
# both faster and much less likely to be rejected/timed-out by the server.
session = requests.Session()
session.headers.update(HEADERS)

# Small delay between requests so we don't hammer the server with thousands
# of rapid-fire requests back to back -- being a "polite" client makes
# timeouts/rejections far less likely.
DELAY_BETWEEN_REQUESTS = 0.15
REQUEST_TIMEOUT = 60  # generous timeout since some responses may be slow


def list_files(url):
    """Parses a PhysioNet directory-listing page and returns links to .psv files
    and to sub-folders (PhysioNet splits training data into subfolders)."""
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    psv_links = []
    folder_links = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href or href.startswith("?") or href == "../":
            continue
        full_url = urljoin(url, href)
        if href.endswith(".psv"):
            psv_links.append(full_url)
        elif href.endswith("/") and full_url.startswith(url):
            folder_links.append(full_url)

    return psv_links, folder_links


def download_file(url, out_path, max_retries=4):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skipped"  # already downloaded, skip (resumable behavior)

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            return "downloaded"
        except requests.RequestException as e:
            wait = 2 * (attempt + 1)
            print(f"  Retry {attempt + 1}/{max_retries} for {url.split('/')[-1]}: {e}")
            print(f"  Waiting {wait}s before retrying...")
            time.sleep(wait)
    print(f"  FAILED after {max_retries} attempts: {url}")
    return "failed"


def crawl_and_download(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    psv_links, folder_links = list_files(url)

    print(f"Found {len(psv_links)} .psv files and {len(folder_links)} sub-folders at {url}")

    downloaded = skipped = failed = 0
    for i, link in enumerate(psv_links):
        filename = link.split("/")[-1]
        out_path = os.path.join(out_dir, filename)
        result = download_file(link, out_path)
        if result == "downloaded":
            downloaded += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(psv_links)} processed "
                  f"(downloaded={downloaded}, skipped={skipped}, failed={failed})")

    print(f"Folder done: {downloaded} downloaded, {skipped} skipped, {failed} failed.")

    for folder_url in folder_links:
        subfolder_name = folder_url.rstrip("/").split("/")[-1]
        crawl_and_download(folder_url, os.path.join(out_dir, subfolder_name))


if __name__ == "__main__":
    print(f"Downloading PhysioNet sepsis training data to: {OUT_DIR}")
    print("This may take a while depending on your connection (thousands of small files).")
    print("Safe to stop (Ctrl+C) and re-run -- already-downloaded files are skipped.\n")
    crawl_and_download(BASE_URL, OUT_DIR)
    print("\nDone.")