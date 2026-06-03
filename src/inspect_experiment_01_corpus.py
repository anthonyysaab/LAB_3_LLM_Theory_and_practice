from pathlib import Path
import pickle

repo_dir = Path(__file__).resolve().parents[1]
text = (repo_dir / "data" / "experiment_01" / "french_poetry_corpus_cleaned.txt").read_text(encoding="utf-8")

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = [stoi[c] for c in text]

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

artifacts_dir = repo_dir / "outputs" / "experiment_01"
artifacts_dir.mkdir(parents=True, exist_ok=True)
with open(artifacts_dir / "vocab.pkl", "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos}, f)

print("Corpus chars:", len(text))
print("Vocab size:", len(chars))
print("Train size:", len(train_data))
print("Val size:", len(val_data))
