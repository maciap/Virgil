from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from captum.influence import TracInCPFast
    _CAPTUM_OK = True
except ImportError:
    _CAPTUM_OK = False


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str           # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None


SUPPORTED_MODELS: Dict[str, str] = {
    "distilbert-base-uncased  (66 M — recommended)": "distilbert-base-uncased",
    "bert-base-uncased        (110 M)":               "bert-base-uncased",
    "roberta-base             (125 M)":               "roberta-base",
    "albert-base-v2           (12 M — fastest)":      "albert-base-v2",
    "distilroberta-base       (82 M)":                "distilroberta-base",
}

_DEFAULT_MODEL_LABEL = "distilbert-base-uncased  (66 M — recommended)"

_DEMO_POS = [
    "This product is absolutely amazing and works perfectly.",
    "I love this, it exceeded all my expectations.",
    "Fantastic quality and fast delivery, very happy.",
    "Best purchase I've made this year, highly recommend.",
    "Great value for money, works exactly as described.",
    "Incredible experience from start to finish.",
    "Would definitely buy this again without hesitation.",
    "The quality is outstanding and delivery was fast.",
    "Exceeded expectations in every single way.",
    "Superb build quality and excellent customer support.",
]

_DEMO_NEG = [
    "Terrible product, broke after one day.",
    "Complete waste of money, very disappointed.",
    "Does not work as advertised, poor quality.",
    "Would not recommend, cheaply made and useless.",
    "Worst purchase ever, returning immediately.",
    "Absolutely dreadful — nothing works as described.",
    "Poor craftsmanship and zero customer support.",
    "Fell apart on first use, utter disappointment.",
    "Misleading description, product is a complete failure.",
    "Cheap materials, stopped working within a week.",
]

_DEMO_EXAMPLES_TEXT = "\n".join(
    [f"1 | {t}" for t in _DEMO_POS] + [f"0 | {t}" for t in _DEMO_NEG]
)

_DEFAULT_TEST_TEXT = "This item feels solid and arrived quickly, I am impressed."


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _dbg_tensor(x, name="x"):
    if torch.is_tensor(x):
        return f"{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    return f"{name}: {type(x)} -> {x}"


class _TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        enc = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]         # (N, L)
        self.attention_mask = enc["attention_mask"]    # (N, L)
        self.labels         = torch.tensor(labels, dtype=torch.long)  # (N,)
        self.texts          = texts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, np.integer):
            idx = int(idx)
        if torch.is_tensor(idx):
            if idx.dim() == 0 or idx.numel() == 1:
                idx = int(idx.item())
            else:
                idx_list = idx.reshape(-1).tolist()
                return (
                    self.input_ids[idx_list],
                    self.attention_mask[idx_list],
                    self.labels[idx_list],
                )
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


class _BERTWrapper(nn.Module):
    def __init__(self, hf_model: AutoModelForSequenceClassification):
        super().__init__()
        self.model = hf_model
        self._fc   = self._find_fc(hf_model)

    @staticmethod
    def _find_fc(hf_model) -> nn.Linear:
        for attr in ("classifier", "cls", "score"):
            layer = getattr(hf_model, attr, None)
            if isinstance(layer, nn.Linear):
                return layer
            if layer is not None:
                for sub_attr in ("out_proj", "dense"):
                    sub = getattr(layer, sub_attr, None)
                    if isinstance(sub, nn.Linear):
                        return sub

        last_linear = None
        for m in hf_model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            return last_linear
        raise RuntimeError("Could not locate a final nn.Linear in the model.")

    @property
    def classifier(self) -> nn.Linear:
        return self._fc

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _fine_tune(
    texts: List[str],
    labels: List[int],
    model_name: str,
    epochs: int,
    max_len: int,
    batch_size: int,
    ckpt_dir: str,
    device: str,
) -> Tuple[_BERTWrapper, Any, List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model  = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model     = _BERTWrapper(hf_model).to(device)

    dataset = _TextDataset(texts, labels, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer   = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_paths: List[str] = []

    for epoch in range(epochs):
        model.train()
        for ids, mask, lbls in loader:
            optimizer.zero_grad()
            logits = model(ids.to(device), mask.to(device))
            loss = loss_fn(logits, lbls.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Save checkpoint WITH LR (Captum expects checkpoints_load_func to return LR)
        lr = float(optimizer.param_groups[0]["lr"])
        path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "lr": lr,
                "epoch": epoch + 1,
            },
            path,
        )
        ckpt_paths.append(path)

    return model, tokenizer, ckpt_paths


def _parse_examples(raw: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    for line in raw.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        label_part, text_part = line.split("|", 1)
        try:
            label = int(label_part.strip())
        except ValueError:
            continue
        if label not in (0, 1):
            continue
        text = text_part.strip()
        if text:
            texts.append(text)
            labels.append(label)
    return texts, labels


class TracInInfluenceClassifier:
    id   = "tracin_influence_classifier"
    name = "Training Data Influence — TracIn (Classifier)"

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_label",
                label="Model",
                type="select",
                options=list(SUPPORTED_MODELS.keys()),
                default=_DEFAULT_MODEL_LABEL,
                help=(
                    "All models are fine-tuned from scratch on your examples. "
                    "albert-base-v2 is the fastest; distilbert-base-uncased gives "
                    "the best speed/quality balance."
                ),
            ),
            FieldSpec(
                key="training_examples",
                label="Training examples  (label | text, one per line)",
                type="textarea",
                default=_DEMO_EXAMPLES_TEXT,
                help=(
                    "Format: 0 or 1 (label), a pipe |, then the text. "
                    "Example:  1 | This product is great. "
                    "Default: 10 positive + 10 negative product reviews."
                ),
            ),
            FieldSpec(
                key="test_text",
                label="Test sentence to explain",
                type="text",
                default=_DEFAULT_TEST_TEXT,
                help="The sentence whose prediction you want to trace back to the training data.",
            ),
            FieldSpec(
                key="epochs",
                label="Fine-tuning epochs (1–8)",
                type="number",
                required=False,
                default=3,
                help=(
                    "One checkpoint is saved per epoch. "
                    "More epochs → more checkpoints → more accurate TracIn scores, "
                    "but longer runtime. Recommended: 3–5."
                ),
            ),
            FieldSpec(
                key="top_k",
                label="Top-K results (1–10)",
                type="number",
                required=False,
                default=5,
                help="Number of proponents and opponents to return.",
            ),
            FieldSpec(
                key="use_gpu",
                label="Use GPU (if available)",
                type="checkbox",
                required=False,
                default=True,
            ),
        ]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not _CAPTUM_OK:
            return {"plugin": self.id, "error": "captum is not installed. Run: pip install captum"}

        MAX_LEN = 128
        BATCH_SIZE = 8

        model_label  = inputs.get("model_label") or _DEFAULT_MODEL_LABEL
        model_name   = SUPPORTED_MODELS.get(model_label, "distilbert-base-uncased")
        raw_examples = inputs.get("training_examples") or _DEMO_EXAMPLES_TEXT
        test_text    = (inputs.get("test_text") or _DEFAULT_TEST_TEXT).strip()
        epochs       = max(1, min(_to_int(inputs.get("epochs", 3), 3), 8))
        top_k        = max(1, min(_to_int(inputs.get("top_k", 5), 5), 10))
        use_gpu      = bool(inputs.get("use_gpu", True))
        device       = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        if not test_text:
            test_text = _DEFAULT_TEST_TEXT

        all_texts, all_labels = _parse_examples(raw_examples)
        if len(all_texts) < 4:
            all_texts  = _DEMO_POS + _DEMO_NEG
            all_labels = [1] * len(_DEMO_POS) + [0] * len(_DEMO_NEG)

        rng = np.random.default_rng(42)
        perm = rng.permutation(len(all_texts))
        all_texts = [all_texts[i] for i in perm]
        all_labels = [all_labels[i] for i in perm]
        n_pos = sum(all_labels)
        n_neg = len(all_labels) - n_pos

        with tempfile.TemporaryDirectory() as ckpt_dir:
            model, tokenizer, ckpt_paths = _fine_tune(
                texts=all_texts,
                labels=all_labels,
                model_name=model_name,
                epochs=epochs,
                max_len=MAX_LEN,
                batch_size=BATCH_SIZE,
                ckpt_dir=ckpt_dir,
                device=device,
            )

            train_dataset = _TextDataset(all_texts, all_labels, tokenizer, max_len=MAX_LEN)

            

            # Captum expects checkpoints_load_func to return a FLOAT learning rate
            def _load_ckpt(m: nn.Module, p: str) -> float:
                ckpt = torch.load(p, map_location="cpu")
                m.load_state_dict(ckpt["model_state_dict"])
                m.to(device)
                return float(ckpt["lr"])

            tracin = TracInCPFast(
                model=model,
                final_fc_layer=model.classifier,
                train_dataset=train_dataset,
                checkpoints=ckpt_paths,
                checkpoints_load_func=_load_ckpt,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                batch_size=BATCH_SIZE,
            )

            model.eval()

            enc = tokenizer(
                test_text,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids  = enc["input_ids"].to(device)         # (1, MAX_LEN)
            mask = enc["attention_mask"].to(device)    # (1, MAX_LEN)

            with torch.no_grad():
                logits = model(ids, mask)              # (1, 2)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_label = int(np.argmax(probs))

            lbl = torch.tensor([pred_label], dtype=torch.long, device=device)  # (1,)
            test_tuple = (ids, mask, lbl)

            

            all_scores = tracin.influence(inputs=test_tuple)

            if torch.is_tensor(all_scores):
                scores_row = all_scores[0].detach().cpu()
            elif hasattr(all_scores, "influence_scores"):
                scores_row = all_scores.influence_scores[0].detach().cpu()
            else:
                raise RuntimeError(f"Unexpected Captum return type: {type(all_scores)}")

            sorted_desc = torch.argsort(scores_row, descending=True)
            prop_indices = sorted_desc[:top_k].tolist()
            prop_score_vals = scores_row[prop_indices].tolist()
            opp_indices = sorted_desc[-top_k:].flip(0).tolist()
            opp_score_vals = scores_row[opp_indices].tolist()

        def _fmt(indices, score_vals) -> List[Dict]:
            out = []
            for rank, (i, s) in enumerate(zip(indices, score_vals), 1):
                i = int(i)
                out.append({
                    "rank":       rank,
                    "text":       all_texts[i],
                    "label":      all_labels[i],
                    "label_name": "positive" if all_labels[i] == 1 else "negative",
                    "score":      float(s),
                })
            return out

        return {
            "plugin":     self.id,
            "model":      model_name,
            "device":     device,
            "n_pos":      n_pos,
            "n_neg":      n_neg,
            "total":      len(all_texts),
            "epochs":     epochs,
            "test_text":  test_text,
            "prediction": {
                "label":      pred_label,
                "label_name": "positive" if pred_label == 1 else "negative",
                "confidence": float(probs[pred_label]),
                "prob_neg":   float(probs[0]),
                "prob_pos":   float(probs[1]),
            },
            "proponents": _fmt(prop_indices, prop_score_vals),
            "opponents":  _fmt(opp_indices, opp_score_vals),
            "params": {
                "model":   model_name,
                "epochs":  epochs,
                "top_k":   top_k,
                "device":  device,
                "n_train": len(all_texts),
                "max_len": MAX_LEN,
            },
        }