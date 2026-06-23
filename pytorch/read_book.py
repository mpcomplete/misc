"""
Strip Project Gutenberg boilerplate from a book .txt file, returning only
the actual book text between the START and END markers.

Usage:
    python3 strip_gutenberg_boilerplate.py input.txt [output.txt]

If output.txt is omitted, the cleaned text is printed to stdout.
"""

import re
import sys
from pathlib import Path

# Matches lines like:
#   *** START OF THIS PROJECT GUTENBERG EBOOK <TITLE> ***
#   *** START OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
#   ***START OF THIS PROJECT GUTENBERG EBOOK***
START_RE = re.compile(
    r"^\s*\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG.*$",
    re.IGNORECASE | re.MULTILINE,
)

END_RE = re.compile(
    r"^\s*\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*$",
    re.IGNORECASE | re.MULTILINE,
)

def extract_book_text(full_text: str) -> str:
    """Return the text strictly between the START and END markers."""
    start_match = START_RE.search(full_text)
    end_match = END_RE.search(full_text)

    if not start_match:
        raise ValueError("Could not find a '*** START ... PROJECT GUTENBERG EBOOK ... ***' marker")
    if not end_match:
        raise ValueError("Could not find a '*** END ... PROJECT GUTENBERG EBOOK ... ***' marker")
    if end_match.start() <= start_match.end():
        raise ValueError("END marker appears before or at START marker")

    body = full_text[start_match.end():end_match.start()]
    return body.strip("\n")

def read_book(input_path: Path) -> str:
    print(f"Processing {input_path}")
    text = input_path.read_text(encoding="utf-8", errors="replace")
    body = extract_book_text(text)
    return body

def read_all_books(input_dir: Path, num_books: int) -> str:
    full_text = ""
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            full_text += read_book(file_path) + "<|endoftext|>"
            num_books -= 1
            if num_books <= 0:
                break
    return full_text

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 strip_gutenberg_boilerplate.py input.txt [output.txt]")
        sys.exit(1)

    try:
        body = read_book(Path(sys.argv[1]))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
        output_path.write_text(body, encoding="utf-8")
        print(f"Wrote cleaned text to {output_path}")
    else:
        print(body[:500])


if __name__ == "__main__":
    main()