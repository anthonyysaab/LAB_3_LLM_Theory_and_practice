from pathlib import Path
import re

inp = Path("corpus/french_poetry_corpus.txt")
out = Path("corpus/french_poetry_corpus_cleaned.txt")

text = inp.read_text(encoding="utf-8")
text = text.replace("\r\n", "\n").replace("\r", "\n")

blocks = re.split(r"(?=<POEM=)", text)
cleaned_blocks = []

for block in blocks:
    block = block.strip()
    if not block:
        continue

    lines = [line.rstrip() for line in block.split("\n")]
    if len(lines) < 3:
        continue

    poem_line = lines[0].strip()
    author_line = lines[1].strip() if lines[1].startswith("<AUTHOR=") else "<AUTHOR=UNKNOWN>"

    poem_match = re.match(r"<POEM=(.*)>", poem_line)
    author_match = re.match(r"<AUTHOR=(.*)>", author_line)

    poem_title = poem_match.group(1).strip() if poem_match else ""
    author_name = author_match.group(1).strip() if author_match else "UNKNOWN"

    body = lines[2:]
    cleaned_body = []

    for line in body:
        s = line.strip()

        if not s:
            cleaned_body.append("")
            continue

        if s == poem_title:
            continue
        if s == author_name:
            continue
        if re.match(r"^.*\(\d{4}.*\)\.?$", s):
            continue
        if re.match(r"^(Sonnet|Ballade|Ode|Élégie|Elegie)\.?$", s, flags=re.IGNORECASE):
            continue

        cleaned_body.append(line)

    cleaned_text = "\n".join(cleaned_body)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    if cleaned_text:
        cleaned_blocks.append(f"{poem_line}\n{author_line}\n\n{cleaned_text}")

final_text = "\n\n".join(cleaned_blocks).strip() + "\n"
out.write_text(final_text, encoding="utf-8")

print("Saved:", out)
print("Characters:", len(final_text))
print("Approx MB:", len(final_text.encode('utf-8')) / (1024 * 1024))
print("Unique chars:", len(set(final_text)))