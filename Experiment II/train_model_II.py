from pathlib import Path
import math
import pickle
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASE OUTPUT FOLDER
# ============================================================
base_dir = Path(r"C:\Users\user\Python\lab3 Experiment II")
artifacts_dir = base_dir / "artifacts"
checkpoints_dir = base_dir / "checkpoints"
samples_dir = base_dir / "samples"

artifacts_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir.mkdir(parents=True, exist_ok=True)
samples_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================
batch_size = 16
block_size = 256
max_iters = 40000
eval_interval = 1000
eval_iters = 32
learning_rate = 3e-4
weight_decay = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"

n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.1

checkpoint_path = checkpoints_dir / "char_gpt_60mb_checkpoint.pt"
best_checkpoint_path = checkpoints_dir / "char_gpt_60mb_best.pt"
final_model_path = checkpoints_dir / "char_gpt_60mb_final.pt"
samples_path = samples_dir / "training_samples_60mb.txt"
final_sample_path = samples_dir / "final_sample_60mb.txt"
training_log_path = artifacts_dir / "training_log_60mb.csv"
vocab_path = artifacts_dir / "vocab_60mb.pkl"

corpus_path = Path(r"C:\Users\user\Python\lab3 Experiment II\french_poetry_corpus_cleaned.txt")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================================================
# LOAD CORPUS
# ============================================================
if not corpus_path.exists():
    raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

text = corpus_path.read_text(encoding="utf-8")

if len(text) <= block_size + 1:
    raise ValueError(
        f"Corpus is too small for block_size={block_size}. "
        f"Need more than {block_size + 1} characters, got {len(text)}."
    )

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s: str):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

with open(vocab_path, "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos}, f)

print(f"Corpus file: {corpus_path}")
print(f"Corpus chars: {len(text):,}")
print(f"Vocab size: {vocab_size}")
print(f"Train size: {len(train_data):,}")
print(f"Val size: {len(val_data):,}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATA LOADER
# ============================================================
def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# MODEL COMPONENTS
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        return idx


# ============================================================
# INIT MODEL + OPTIMIZER + SCALER
# ============================================================
model = CharGPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params / 1e6:.2f}M")

current_config = {
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
    "eval_interval": eval_interval,
    "eval_iters": eval_iters,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "vocab_size": vocab_size,
    "corpus_path": str(corpus_path),
}


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================
# LOGGING HELPERS
# ============================================================
def append_log(step, train_loss, val_loss, train_bpc, val_bpc):
    file_exists = training_log_path.exists()
    with open(training_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "train_loss", "val_loss", "train_bpc", "val_bpc"])
        writer.writerow([step, train_loss, val_loss, train_bpc, val_bpc])


def save_checkpoint(step, best_val_loss=None):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": current_config,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def save_best_checkpoint(step, best_val_loss):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": current_config,
            "best_val_loss": best_val_loss,
        },
        best_checkpoint_path,
    )


# ============================================================
# RESUME FROM CHECKPOINT IF IT EXISTS
# ============================================================
start_step = 0
best_val_loss = float("inf")

if checkpoint_path.exists():
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_config = ckpt.get("config", {})
    compatible = True

    important_keys = ["block_size", "n_embd", "n_head", "n_layer", "vocab_size", "corpus_path"]
    for key in important_keys:
        if saved_config.get(key) != current_config.get(key):
            compatible = False
            break

    if compatible:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                print("Warning: could not load AMP scaler state, continuing.")

        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resuming from step {start_step}")
    else:
        print("WARNING: checkpoint is incompatible with current run.")
        print("Checkpoint config:", saved_config)
        print("Current config   :", current_config)
        print("Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")


# ============================================================
# SAMPLE PROMPT HELPER
# ============================================================
def make_prompt_tensor(prompt="L"):
    safe_prompt = "".join(ch for ch in prompt if ch in stoi)
    if not safe_prompt:
        safe_prompt = text[:1]
    return torch.tensor([encode(safe_prompt)], dtype=torch.long, device=device), safe_prompt


# ============================================================
# TRAINING LOOP
# ============================================================
model.train()
t0 = time.time()

for step in range(start_step, max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        train_bpc = losses["train"] / math.log(2)
        val_bpc = losses["val"] / math.log(2)

        print(
            f"step {step:5d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | "
            f"train bpc {train_bpc:.4f} | "
            f"val bpc {val_bpc:.4f}"
        )

        context, used_prompt = make_prompt_tensor("L")

        sample = decode(
            model.generate(
                context,
                max_new_tokens=500,
                temperature=0.9,
                top_k=50,
            )[0].tolist()
        )

        print("\n--- SAMPLE ---\n")
        print(sample)
        print("\n--------------\n")

        with open(samples_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n{'=' * 80}\n"
                f"STEP {step}\n"
                f"prompt={used_prompt}\n"
                f"train_loss={losses['train']:.6f} "
                f"val_loss={losses['val']:.6f} "
                f"train_bpc={train_bpc:.6f} "
                f"val_bpc={val_bpc:.6f}\n\n"
                f"{sample}\n"
            )

        append_log(step, losses["train"], losses["val"], train_bpc, val_bpc)

        save_checkpoint(step, best_val_loss=best_val_loss)
        print(f"Checkpoint saved at step {step}")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_best_checkpoint(step, best_val_loss)
            print(f"New best checkpoint saved at step {step} (val_loss={best_val_loss:.6f})")

    xb, yb = get_batch("train")

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=use_amp):
        _, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()


# ============================================================
# SAVE FINAL MODEL + FINAL SAMPLE
# ============================================================
save_checkpoint(max_iters - 1, best_val_loss=best_val_loss)
torch.save(model.state_dict(), final_model_path)

context, used_prompt = make_prompt_tensor("L")
final_sample = decode(
    model.generate(
        context,
        max_new_tokens=800,
        temperature=0.9,
        top_k=50,
    )[0].tolist()
)

with open(final_sample_path, "w", encoding="utf-8") as f:
    f.write(final_sample)

elapsed = time.time() - t0

print(f"Final model saved to: {final_model_path}")
print(f"Best checkpoint saved to: {best_checkpoint_path}")
print(f"Training samples saved to: {samples_path}")
print(f"Final sample saved to: {final_sample_path}")
print(f"Training log saved to: {training_log_path}")
print(f"Total runtime: {elapsed / 60:.2f} minutes")