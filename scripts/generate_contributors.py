"""
Script para mantener actualizado el CONTRIBUTORS.md automáticamente.

:authors: Tomás Macrade
:date: 01/03/2026
"""

import re
from pathlib import Path

ROOT = Path("whiteboxml")
OUTPUT_FILE = Path("CONTRIBUTORS.md")

author_pattern = re.compile(r":authors:\s*(.+)")

authors = set()

for path in ROOT.rglob("*.py"):
    content = path.read_text(encoding="utf-8")
    matches = author_pattern.findall(content)

    for match in matches:
        split_authors = match.split(",")

        for author in split_authors:
            clean_author = author.strip()
            if clean_author:
                authors.add(clean_author)

authors = sorted(authors)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    f.write("# Contributors\n\n")
    for author in authors:
        f.write(f"- {author}\n")

print("CONTRIBUTORS.md generated successfully.")