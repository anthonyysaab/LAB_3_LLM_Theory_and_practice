from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(r"C:\Users\user\Python\lab3 Experiment II\corpus_vers.csv")
OUTPUT_FILE = Path(r"C:\Users\user\Python\lab3 Experiment II\french_poetry_corpus_cleaned.txt")
BAD_ROWS_FILE = Path(r"C:\Users\user\Python\lab3 Experiment II\bad_rows_log.tsv")


# ============================================================
# CLEANING
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u00A0", " ")

    for ch in ["\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"]:
        text = text.replace(ch, "")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " ")
    text = " ".join(text.split()).strip()

    return text


def detect_dialect(file_path: Path, encoding: str = "utf-8-sig") -> csv.Dialect:
    with file_path.open("r", encoding=encoding, newline="") as f:
        sample = f.read(10000)
    return csv.Sniffer().sniff(sample, delimiters=",;\t|")


# ============================================================
# MAIN
# ============================================================

def build_corpus() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written_verses = 0
    skipped_empty = 0
    duplicate_ids = 0
    bad_rows = 0

    current_poem = None
    last_id_vers = None

    dialect = detect_dialect(INPUT_FILE)

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as fin, \
         OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as fout, \
         BAD_ROWS_FILE.open("w", encoding="utf-8", newline="") as fbad:

        reader = csv.DictReader(fin, dialect=dialect)
        bad_writer = csv.writer(fbad, delimiter="\t")
        bad_writer.writerow(["row_number", "reason", "raw_row"])

        expected = {"ID_POEME", "NUM_ABS", "VERS"}
        found = set(reader.fieldnames or [])
        missing = expected - found
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}\n"
                f"Found columns: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            total_rows += 1

            try:
                id_poeme = normalize_text(row.get("ID_POEME", ""))
                id_vers = normalize_text(row.get("ID_VERS", ""))
                vers = normalize_text(row.get("VERS", ""))

                if id_vers and id_vers == last_id_vers:
                    duplicate_ids += 1
                    continue
                last_id_vers = id_vers

                if not vers:
                    skipped_empty += 1
                    continue

                if current_poem is None:
                    current_poem = id_poeme
                elif id_poeme != current_poem:
                    fout.write("\n")
                    current_poem = id_poeme

                fout.write(vers + "\n")
                written_verses += 1

            except Exception as e:
                bad_rows += 1
                bad_writer.writerow([row_num, str(e), repr(row)])

    print("Done.")
    print(f"Input file      : {INPUT_FILE}")
    print(f"Output corpus   : {OUTPUT_FILE}")
    print(f"Bad rows log    : {BAD_ROWS_FILE}")
    print(f"Total rows      : {total_rows}")
    print(f"Written verses  : {written_verses}")
    print(f"Skipped empty   : {skipped_empty}")
    print(f"Duplicate IDs   : {duplicate_ids}")
    print(f"Bad rows        : {bad_rows}")


if __name__ == "__main__":
    build_corpus()