# LLM Visualize

A small, inspectable debugger for the GPT model built in Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

The goal is not to render every multiplication. It is to move through one real forward pass and inspect the tensors that matter:

```
tokens -> embeddings -> layer norm -> Q/K/V -> QK^T -> softmax
       -> attention output -> residual -> layer norm
       -> W1 -> GELU -> W2 -> residual -> logits
```

## Train once, visualize many times

Do **not** retrain the model whenever HTML/CSS/JS changes.

After training, save the checkpoint:

```python
from python.trace_raschka import save_model_checkpoint

save_model_checkpoint(
    model,
    "/kaggle/working/verdict-gpt.pth",
    config=GPT_CONFIG_124M,
    extra={"dataset": "the-verdict.txt"},
)
```

Later, reconstruct the same model class/config and load the weights:

```python
from python.trace_raschka import load_model_checkpoint

load_model_checkpoint(model, "/kaggle/working/verdict-gpt.pth", device="cpu")
model.eval()
```

On Kaggle, keep the checkpoint as a notebook output when you Save Version, or publish it as a Kaggle Dataset so future sessions can attach it without retraining.

## Create a trace

The tracer expects the Raschka-style attributes used in the book:
`tok_emb`, `pos_emb`, `drop_emb`, `trf_blocks`, `final_norm`, and `out_head`.

```python
import torch
import tiktoken
from python.trace_raschka import trace_forward_pass, write_trace

tokenizer = tiktoken.get_encoding("gpt2")
text = "Every effort moves you"
ids = tokenizer.encode(text)
input_ids = torch.tensor(ids).unsqueeze(0)

trace = trace_forward_pass(
    model,
    input_ids,
    tokens=[tokenizer.decode([i]) for i in ids],
    decode_token=lambda i: tokenizer.decode([i]),
    top_k=10,
)

write_trace(trace, "/kaggle/working/trace.json")
```

The trace is inference only: `model.eval()` + `torch.no_grad()`. It does not train.

## Kaggle: one self-contained HTML file

No Node or TypeScript build step is required.

```python
from python.export_visualizer import build_visualizer

build_visualizer(
    "/kaggle/working/trace.json",
    "/kaggle/working/llm-debugger.html",
)
```

Display it in the notebook:

```python
from IPython.display import HTML, display
display(HTML(open("/kaggle/working/llm-debugger.html", encoding="utf-8").read()))
```

If a Kaggle rendering mode sanitizes notebook JavaScript, open/download the generated `llm-debugger.html` from notebook outputs. It is standalone.

## Views

- **Overview** — the forward-pass pipeline.
- **Tokens** — token text and IDs.
- **Embeddings** — token, position, and combined vectors as compact heat rows.
- **Attention** — choose layer/head, inspect the attention matrix, click a cell to expand its Q·K calculation and softmax weight.
- **MLP / GELU** — W1 output, GELU output, and W2 output for one token.
- **Logits** — top next-token probabilities.

The browser is plain HTML/CSS/JavaScript. The real calculations remain in PyTorch.
