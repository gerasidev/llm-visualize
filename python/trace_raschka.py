"""Forward-pass tracing helpers for Raschka-style GPT models.

This module intentionally does not own the model definition or training loop.
It accepts an already-created model, optionally loads a checkpoint into it,
and records the tensors useful for an educational debugger.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch


def _clean(value: Any) -> Any:
    """Convert tensors/numbers to strict JSON-safe Python values."""
    if isinstance(value, torch.Tensor):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


def save_model_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    config: Optional[dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Save trained weights once so visualization iterations do not retrain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)
    return path


def load_model_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """Load either this project's checkpoint payload or a raw state_dict."""
    payload = torch.load(Path(path), map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = {
            "config": payload.get("config", {}),
            "extra": payload.get("extra", {}),
        }
    else:
        state_dict = payload
        metadata = {"config": {}, "extra": {}}

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return metadata


def _require_attrs(obj: Any, names: Iterable[str], label: str) -> None:
    missing = [name for name in names if not hasattr(obj, name)]
    if missing:
        raise AttributeError(
            f"{label} is missing expected Raschka-style attributes: {missing}"
        )


def _trace_attention(att: torch.nn.Module, x: torch.Tensor) -> Dict[str, Any]:
    _require_attrs(
        att,
        ["W_query", "W_key", "W_value", "out_proj", "num_heads", "head_dim", "mask"],
        "attention module",
    )

    batch_size, num_tokens, d_out = x.shape
    if batch_size != 1:
        raise ValueError("The visualizer currently traces one prompt at a time (batch=1).")

    keys = att.W_key(x)
    queries = att.W_query(x)
    values = att.W_value(x)

    keys_h = keys.view(batch_size, num_tokens, att.num_heads, att.head_dim).transpose(1, 2)
    queries_h = queries.view(batch_size, num_tokens, att.num_heads, att.head_dim).transpose(1, 2)
    values_h = values.view(batch_size, num_tokens, att.num_heads, att.head_dim).transpose(1, 2)

    raw_scores = queries_h @ keys_h.transpose(2, 3)

    causal_mask = att.mask.bool()[:num_tokens, :num_tokens]
    masked_scores = raw_scores.masked_fill(causal_mask, float("-inf"))
    scaled_scores = masked_scores / (att.head_dim ** 0.5)
    weights = torch.softmax(scaled_scores, dim=-1)

    weights_for_context = att.dropout(weights) if hasattr(att, "dropout") else weights
    context_h = weights_for_context @ values_h
    context = (
        context_h.transpose(1, 2)
        .contiguous()
        .view(batch_size, num_tokens, d_out)
    )
    output = att.out_proj(context)

    return {
        "num_heads": int(att.num_heads),
        "head_dim": int(att.head_dim),
        "q": _clean(queries_h[0]),
        "k": _clean(keys_h[0]),
        "v": _clean(values_h[0]),
        "raw_scores": _clean(raw_scores[0]),
        "scaled_scores": _clean(scaled_scores[0]),
        "weights": _clean(weights[0]),
        "context_by_head": _clean(context_h[0]),
        "context_combined": _clean(context[0]),
        "output": _clean(output[0]),
    }


def _trace_mlp(ff: torch.nn.Module, x: torch.Tensor) -> Dict[str, Any]:
    _require_attrs(ff, ["layers"], "feed-forward module")
    layers = ff.layers
    if len(layers) < 3:
        raise ValueError("Expected ff.layers = Linear -> GELU -> Linear.")

    linear1 = layers[0](x)
    activated = layers[1](linear1)
    linear2 = layers[2](activated)

    return {
        "input": _clean(x[0]),
        "w1_output": _clean(linear1[0]),
        "gelu_output": _clean(activated[0]),
        "w2_output": _clean(linear2[0]),
        "hidden_dim": int(linear1.shape[-1]),
    }


def trace_forward_pass(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    tokens: Optional[list[str]] = None,
    decode_token: Optional[Callable[[int], str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Record one deterministic forward pass without changing learned weights."""
    _require_attrs(
        model,
        ["tok_emb", "pos_emb", "drop_emb", "trf_blocks", "final_norm", "out_head"],
        "model",
    )

    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("token_ids must have shape [tokens] or [1, tokens].")

    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    num_tokens = int(token_ids.shape[1])

    if tokens is None:
        tokens = [str(int(i)) for i in token_ids[0].detach().cpu().tolist()]
    if len(tokens) != num_tokens:
        raise ValueError("tokens length must match token_ids length.")

    was_training = model.training
    model.eval()

    with torch.no_grad():
        tok_embeds = model.tok_emb(token_ids)
        positions = torch.arange(num_tokens, device=device)
        pos_embeds = model.pos_emb(positions)
        x = tok_embeds + pos_embeds
        x = model.drop_emb(x)

        trace: Dict[str, Any] = {
            "schema_version": 1,
            "metadata": {
                "model_class": model.__class__.__name__,
                "num_layers": len(model.trf_blocks),
                "embedding_dim": int(tok_embeds.shape[-1]),
                "sequence_length": num_tokens,
                "top_k": int(top_k),
            },
            "tokens": [
                {"index": i, "text": tokens[i], "id": int(token_ids[0, i].item())}
                for i in range(num_tokens)
            ],
            "embeddings": {
                "token": _clean(tok_embeds[0]),
                "position": _clean(pos_embeds),
                "combined": _clean(x[0]),
            },
            "layers": [],
        }

        for layer_index, block in enumerate(model.trf_blocks):
            _require_attrs(
                block,
                ["norm1", "att", "norm2", "ff", "drop_shortcut"],
                f"transformer block {layer_index}",
            )

            block_input = x
            norm1_out = block.norm1(block_input)
            attention = _trace_attention(block.att, norm1_out)

            # Use the real module output as the source of truth.
            attention_out = block.att(norm1_out)
            after_attention = block_input + block.drop_shortcut(attention_out)

            norm2_out = block.norm2(after_attention)
            mlp = _trace_mlp(block.ff, norm2_out)
            ff_out = block.ff(norm2_out)
            after_mlp = after_attention + block.drop_shortcut(ff_out)

            trace["layers"].append(
                {
                    "index": layer_index,
                    "input": _clean(block_input[0]),
                    "norm1_output": _clean(norm1_out[0]),
                    "attention": attention,
                    "after_attention_residual": _clean(after_attention[0]),
                    "norm2_output": _clean(norm2_out[0]),
                    "mlp": mlp,
                    "output": _clean(after_mlp[0]),
                }
            )

            x = after_mlp

        final_norm = model.final_norm(x)
        logits = model.out_head(final_norm)

        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        k = min(int(top_k), int(probs.shape[-1]))
        top_probs, top_ids = torch.topk(probs, k=k)

        predictions = []
        for rank, (prob, token_id) in enumerate(zip(top_probs, top_ids), start=1):
            tid = int(token_id.item())
            predictions.append(
                {
                    "rank": rank,
                    "token_id": tid,
                    "token": decode_token(tid) if decode_token else str(tid),
                    "probability": round(float(prob.item()), 8),
                    "logit": round(float(last_logits[tid].item()), 6),
                }
            )

        trace["final"] = {
            "norm": _clean(final_norm[0]),
            "last_token_predictions": predictions,
        }

    if was_training:
        model.train()

    return trace


def write_trace(trace: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return path
