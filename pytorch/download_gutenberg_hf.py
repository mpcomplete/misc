#!/usr/bin/env python3
"""
Download the English subset of the Project Gutenberg dataset from Hugging Face
(manu/project_gutenberg) and save each book as an individual .txt file.

Requires: pip install datasets --break-system-packages
"""

import re
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("gutenberg_txt")
DATASET_NAME = "manu/project_gutenberg"
SPLIT = "en"


def safe_filename(book_id: str) -> str:
    """Sanitize the book id for use as a filename."""
    return re.sub(r"[^\w\-]", "_", book_id)


def extract_title(text: str, fallback: str) -> str:
    """Try to pull a short title from the first line for nicer filenames."""
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    match = re.search(r"Project Gutenberg.*?of (.+?)(?:, by|\n)", first_line, re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        title = re.sub(r"[^\w\- ]", "", title)[:80].strip().replace(" ", "_")
        if title:
            return f"{fallback}_{title}"
    return fallback


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Saving books to: {OUTPUT_DIR.resolve()}\n")

    # streaming=True avoids loading the whole ~14GB dataset into memory at once
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    count = 0
    skipped = 0
    for row in ds:
        book_id = row.get("id", f"unknown_{count}")
        text = row.get("text", "")

        if not text:
            skipped += 1
            continue

        fname = safe_filename(extract_title(text, book_id)) + ".txt"
        dest = OUTPUT_DIR / fname

        if dest.exists():
            count += 1
            continue

        dest.write_text(text, encoding="utf-8")
        count += 1

        if count % 500 == 0:
            print(f"  ...{count} books saved so far")

    print(f"\nDone. Saved {count} books, skipped {skipped} empty rows.")


if __name__ == "__main__":
    main()