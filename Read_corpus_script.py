from pathlib import Path
import pickle

text = Path("corpus/french_poetry_corpus_cleaned.txt").read_text(encoding="utf-8")

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = [stoi[c] for c in text]

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

Path("artifacts").mkdir(exist_ok=True)
with open("artifacts/vocab.pkl", "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos}, f)

print("Corpus chars:", len(text))
print("Vocab size:", len(chars))
print("Train size:", len(train_data))
print("Val size:", len(val_data))