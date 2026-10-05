# LLM Visualize

An educational debugger for one real forward pass through the tiny **Mini-World GPT** trained from scratch in the Kaggle notebook.

The purpose is to make Transformer internals inspectable rather than to build a production LLM UI.

## Run locally

Requirements: Node.js 20+.

```bash
git clone https://github.com/gerasidev/llm-visualize.git
cd llm-visualize
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`).

### Load the Mini-World trace

The browser needs the forward-pass JSON, not the PyTorch checkpoint.

You have two options:

1. **Simplest:** start the app with `npm run dev`, then drag `mini-world-trace.json` into the page.
2. **Automatic:** copy the file to:

```text
frontend/mini-world-trace.json
```

Then `npm run dev` loads it automatically.

The trace and model checkpoint are intentionally ignored by Git. Keep `mini-world-gpt.pth` as your trained-model artifact; it is only needed when you want Python/PyTorch to generate a new trace for a different prompt.

## What the debugger shows

The current local UI walks through:

```text
tokens
  -> token + position embeddings
  -> self-attention (layer/head)
  -> Q · K / sqrt(d) drill-down
  -> softmax attention weights
  -> residual stream
  -> W1 -> GELU -> W2
  -> final logits / next-token probabilities
```

Important views:

- **Overview** — the complete inference path.
- **Embeddings** — choose a token and inspect token + position = combined representation.
- **Attention** — choose any of the 4 layers and 4 heads; click a heatmap cell to inspect the actual Q·K calculation behind that attention weight.
- **Residual stream** — see what attention and the MLP add to a selected token representation.
- **MLP / GELU** — inspect the strongest hidden activations for a token.
- **Prediction** — inspect the top next-token probabilities and logits.

The visualizer uses the tensors from `mini-world-trace.json`. Changing HTML/CSS/JavaScript does **not** retrain the model.

## Training notebook

Notebook:

```text
notebooks/kaggle_mini_world_visualizer.ipynb
```

The notebook trains the small GPT on `mini-world.txt`, saves `mini-world-gpt.pth`, traces one forward pass, and exports `mini-world-trace.json`.

The model used for the current trace has:

- 4 Transformer layers
- 4 attention heads
- embedding dimension 128
- context length 32

## Standalone Kaggle export

The Python exporter still works:

```python
from python.export_visualizer import build_visualizer

build_visualizer(
    "/kaggle/working/mini-world-trace.json",
    "/kaggle/working/mini-world-debugger.html",
)
```

It embeds the same frontend and trace into one standalone HTML file.
