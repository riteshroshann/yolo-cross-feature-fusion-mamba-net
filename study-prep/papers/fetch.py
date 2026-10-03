"""Download the study papers listed in index.csv from their original hosts, skipping files already present."""
import csv
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
UA = {"User-Agent": "cffm-net-study-fetch/1.0"}


def pdf_url(source):
    return source.replace("/abs/", "/pdf/") if "arxiv.org/abs/" in source else source


def main():
    rows = list(csv.DictReader(open(HERE / "index.csv", encoding="utf-8")))
    for r in rows:
        out = HERE / r["file"]
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(urllib.request.Request(pdf_url(r["source"]), headers=UA), timeout=60).read()
        if not data.startswith(b"%PDF"):
            print("not a PDF, skipped:", r["file"])
            continue
        out.write_bytes(data)
        print("fetched", r["file"])
        time.sleep(3)
    print(f"{sum((HERE / r['file']).exists() for r in rows)} of {len(rows)} papers present")


if __name__ == "__main__":
    main()
