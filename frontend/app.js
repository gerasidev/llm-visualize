(() => {
  const state = {
    trace: null,
    step: "overview",
    layer: 0,
    head: 0,
    token: 0,
  };

  const el = {
    content: document.getElementById("content"),
    empty: document.getElementById("emptyState"),
    summary: document.getElementById("traceSummary"),
    selectors: document.getElementById("selectors"),
    layer: document.getElementById("layerSelect"),
    head: document.getElementById("headSelect"),
    file: document.getElementById("traceFile"),
    nav: document.getElementById("stepNav"),
  };

  const esc = (value) => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

  const fmt = (value, digits = 4) => {
    if (value === null || value === undefined) return "—";
    if (!Number.isFinite(Number(value))) return "—";
    return Number(value).toFixed(digits);
  };

  function maxAbs(values) {
    let m = 0;
    for (const v of values.flat(Infinity)) {
      if (v !== null && Number.isFinite(Number(v))) {
        m = Math.max(m, Math.abs(Number(v)));
      }
    }
    return m || 1;
  }

  function heatStyle(value, scale) {
    if (value === null || value === undefined) return "background:#f0f1f3";
    const x = Math.max(-1, Math.min(1, Number(value) / scale));
    const alpha = 0.12 + 0.72 * Math.abs(x);
    return x >= 0
      ? `background:rgba(49,87,213,${alpha})`
      : `background:rgba(190,65,82,${alpha})`;
  }

  function vectorStrip(values, limit = 18) {
    const shown = values.slice(0, limit);
    return `<div class="value-strip">${shown.map(v =>
      `<span class="value-pill" title="${fmt(v, 6)}">${fmt(v, 2)}</span>`
    ).join("")}${values.length > limit ? '<span class="value-pill">…</span>' : ""}</div>`;
  }

  function populateSelectors() {
    el.layer.innerHTML = state.trace.layers.map((_, i) =>
      `<option value="${i}">Layer ${i + 1}</option>`
    ).join("");
    state.layer = Math.min(state.layer, state.trace.layers.length - 1);
    el.layer.value = String(state.layer);
    populateHeads();
  }

  function populateHeads() {
    const att = state.trace.layers[state.layer].attention;
    el.head.innerHTML = Array.from({ length: att.num_heads }, (_, i) =>
      `<option value="${i}">Head ${i + 1}</option>`
    ).join("");
    state.head = Math.min(state.head, att.num_heads - 1);
    el.head.value = String(state.head);
  }

  function setTrace(trace) {
    if (!trace || !Array.isArray(trace.layers) || !Array.isArray(trace.tokens)) {
      throw new Error("This does not look like an LLM Visualize trace.");
    }
    state.trace = trace;
    state.layer = 0;
    state.head = 0;
    state.token = 0;

    const m = trace.metadata || {};
    el.summary.textContent =
      `${m.model_class || "GPT"} · ${m.num_layers ?? trace.layers.length} layers · ${trace.tokens.length} tokens`;

    el.empty.classList.add("hidden");
    el.content.classList.remove("hidden");
    el.selectors.classList.remove("hidden");
    populateSelectors();
    render();
  }

  function title(text, subtitle) {
    return `<h2 class="page-title">${text}</h2><p class="page-subtitle">${subtitle}</p>`;
  }

  function renderOverview() {
    const m = state.trace.metadata;
    const nodes = [
      "Tokens", "Embeddings", "LayerNorm", "Q / K / V", "QKᵀ", "Softmax",
      "Attention output", "Residual", "LayerNorm", "W1", "GELU", "W2",
      "Residual", "Logits"
    ];

    return title(
      "One forward pass",
      "The learned weights stay fixed. These activations are produced by this prompt."
    ) + `
      <div class="card">
        <div class="pipeline">
          ${nodes.map((node, i) =>
            `<span class="pipe-node">${node}</span>${i < nodes.length - 1 ? '<span class="arrow">→</span>' : ''}`
          ).join("")}
        </div>
      </div>
      <div class="grid three" style="margin-top:16px">
        <div class="card"><h3>Prompt</h3><p>${state.trace.tokens.map(t => esc(t.text)).join(" ")}</p></div>
        <div class="card"><h3>Model shape</h3><p>${m.num_layers} layers · embedding dim ${m.embedding_dim} · sequence length ${m.sequence_length}</p></div>
        <div class="card"><h3>Debugger rule</h3><p>Show summaries first. Expand one calculation only when it helps explain a number.</p></div>
      </div>
      <div class="note" style="margin-top:16px">The trace is inference only. Editing this UI never requires retraining the model.</div>
    `;
  }

  function renderTokens() {
    return title(
      "Tokens",
      "The tokenizer maps text pieces to token IDs. The embedding table then looks up one learned vector per ID."
    ) + `
      <div class="card"><div class="token-row">
        ${state.trace.tokens.map(t => `
          <div class="token-chip">
            <strong>${esc(t.text || "␠")}</strong>
            <span>ID ${t.id} · position ${t.index}</span>
          </div>
        `).join("")}
      </div></div>
    `;
  }

  function renderEmbeddingMatrix(name, matrix) {
    const scale = maxAbs(matrix);
    const dims = Math.min(20, matrix[0]?.length || 0);

    return `<div class="card">
      <h3>${name}</h3>
      <div class="matrix-wrap"><table class="vector-table"><tbody>
        ${matrix.map((row, i) => `<tr>
          <td>${esc(state.trace.tokens[i]?.text ?? i)}</td>
          <td><div class="vector-cells">${row.slice(0, dims).map((v, d) =>
            `<span class="vector-cell" style="${heatStyle(v, scale)}" title="dim ${d}: ${fmt(v, 6)}"></span>`
          ).join("")}</div></td>
        </tr>`).join("")}
      </tbody></table></div>
      <p>Showing the first ${dims} dimensions. Hover a square for its real value.</p>
    </div>`;
  }

  function renderEmbeddings() {
    const e = state.trace.embeddings;
    return title(
      "Embeddings",
      "Token embeddings are learned rows. Position embeddings are added before the first Transformer block."
    ) + `
      <div class="grid">
        ${renderEmbeddingMatrix("Token embedding", e.token)}
        ${renderEmbeddingMatrix("Position embedding", e.position)}
        ${renderEmbeddingMatrix("Combined input", e.combined)}
      </div>
    `;
  }

  function attentionCellMath(i, j) {
    const att = state.trace.layers[state.layer].attention;
    const q = att.q[state.head][i];
    const k = att.k[state.head][j];
    const products = q.map((v, d) => Number(v) * Number(k[d]));
    const dot = products.reduce((a, b) => a + b, 0);
    const scaled = dot / Math.sqrt(att.head_dim);

    const row = att.scaled_scores[state.head][i];
    const finite = row.filter(v => v !== null).map(Number);
    const rowMax = Math.max(...finite);
    const denom = finite.reduce((sum, v) => sum + Math.exp(v - rowMax), 0);
    const weight = att.weights[state.head][i][j];
    const numerator = Math.exp(scaled - rowMax);

    const tokenI = esc(state.trace.tokens[i].text);
    const tokenJ = esc(state.trace.tokens[j].text);
    const shown = products.slice(0, 12).map((p, d) =>
      `${fmt(q[d],3)}×${fmt(k[d],3)}=${fmt(p,3)}`
    ).join(" + ");

    return `
      <div id="attentionMath" class="math-box">
        <div class="math-line"><strong>Query token:</strong> ${tokenI} &nbsp; <strong>Key token:</strong> ${tokenJ}</div>
        <div class="math-line">Q · K = ${fmt(dot, 6)}</div>
        <div class="math-products">${shown}${products.length > 12 ? " + … (" + products.length + " dimensions total)" : ""}</div>
        <div class="math-line">scaled score = ${fmt(dot,6)} / √${att.head_dim} = <strong>${fmt(scaled,6)}</strong></div>
        <div class="math-line">softmax = exp(score − rowMax) / Σ exp(all allowed scores − rowMax)</div>
        <div class="math-line">= ${fmt(numerator,6)} / ${fmt(denom,6)} = <strong>${fmt(weight,6)}</strong></div>
      </div>`;
  }

  function renderAttention() {
    const att = state.trace.layers[state.layer].attention;
    const weights = att.weights[state.head];
    const scale = Math.max(...weights.flat().map(Number), 1e-12);
    const tokens = state.trace.tokens;

    return title(
      "Self-attention",
      "Choose a layer and head. Click an unmasked matrix cell to see exactly where that number came from."
    ) + `
      <div class="grid two">
        <div class="card">
          <h3>Attention weights · Layer ${state.layer + 1} · Head ${state.head + 1}</h3>
          <div class="matrix-wrap">
            <table class="matrix">
              <thead><tr><th>Q \ K</th>${tokens.map(t => `<th>${esc(t.text)}</th>`).join("")}</tr></thead>
              <tbody>
                ${weights.map((row, i) => `<tr><th>${esc(tokens[i].text)}</th>
                  ${row.map((v, j) => {
                    const masked = att.scaled_scores[state.head][i][j] === null;
                    if (masked) return '<td><button class="masked" disabled>×</button></td>';
                    return `<td><button class="att-cell" data-i="${i}" data-j="${j}" style="${heatStyle(v, scale)}">${fmt(v,3)}</button></td>`;
                  }).join("")}
                </tr>`).join("")}
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <h3>What you are looking at</h3>
          <p>Each row is one query token. Each column is one key token. The value is the softmax-normalized attention weight for this head.</p>
          <p><strong>Q, K and V are activations.</strong> They were produced by multiplying current hidden states by learned Wq, Wk and Wv matrices.</p>
          <div id="attentionMath" class="note">Click an attention cell.</div>
        </div>
      </div>

      <div class="grid three" style="margin-top:16px">
        <div class="card"><h3>Q · first token</h3>${vectorStrip(att.q[state.head][0])}</div>
        <div class="card"><h3>K · first token</h3>${vectorStrip(att.k[state.head][0])}</div>
        <div class="card"><h3>V · first token</h3>${vectorStrip(att.v[state.head][0])}</div>
      </div>
    `;
  }

  function renderMLP() {
    const layer = state.trace.layers[state.layer];
    const mlp = layer.mlp;
    const tokenIndex = Math.min(state.token, state.trace.tokens.length - 1);

    return title(
      "MLP and GELU",
      "The feed-forward network transforms each token position independently after attention mixes information between tokens."
    ) + `
      <div class="card">
        <label>Token
          <select id="tokenSelect" class="token-select">
            ${state.trace.tokens.map((tok, i) =>
              `<option value="${i}" ${i === tokenIndex ? "selected" : ""}>${esc(tok.text)} · pos ${i}</option>`
            ).join("")}
          </select>
        </label>

        <div style="margin-top:18px">
          <div class="stat-row"><div class="stat-label">MLP input</div><div>${vectorStrip(mlp.input[tokenIndex])}</div></div>
          <div class="stat-row"><div class="stat-label">W1 output</div><div>${vectorStrip(mlp.w1_output[tokenIndex])}</div></div>
          <div class="stat-row"><div class="stat-label">GELU output</div><div>${vectorStrip(mlp.gelu_output[tokenIndex])}</div></div>
          <div class="stat-row"><div class="stat-label">W2 output</div><div>${vectorStrip(mlp.w2_output[tokenIndex])}</div></div>
        </div>
      </div>

      <div class="note" style="margin-top:16px">
        W1 and W2 are learned parameters. The rows above are temporary activations for this prompt. Hidden MLP width: ${mlp.hidden_dim}.
      </div>
    `;
  }

  function renderLogits() {
    const preds = state.trace.final.last_token_predictions || [];
    const max = Math.max(...preds.map(p => Number(p.probability)), 1e-12);

    return title(
      "Next-token prediction",
      "The final hidden state is projected into vocabulary logits. Softmax converts those logits into probabilities."
    ) + `
      <div class="card">
        <div class="predictions">
          ${preds.map(p => `
            <div class="prediction">
              <div class="rank">#${p.rank}</div>
              <div><strong>${esc(p.token)}</strong><br><span class="muted">ID ${p.token_id}</span></div>
              <div class="bar-track"><div class="bar" style="width:${Math.max(1, Number(p.probability) / max * 100)}%"></div></div>
              <div class="number">${(Number(p.probability) * 100).toFixed(4)}%</div>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }

  function render() {
    if (!state.trace) return;

    const showSelectors = ["attention", "mlp"].includes(state.step);
    el.selectors.classList.toggle("hidden", !showSelectors);

    const renderers = {
      overview: renderOverview,
      tokens: renderTokens,
      embeddings: renderEmbeddings,
      attention: renderAttention,
      mlp: renderMLP,
      logits: renderLogits,
    };

    el.content.innerHTML = renderers[state.step]();

    if (state.step === "attention") {
      document.querySelectorAll(".att-cell").forEach(button => {
        button.addEventListener("click", () => {
          const i = Number(button.dataset.i);
          const j = Number(button.dataset.j);
          document.getElementById("attentionMath").outerHTML = attentionCellMath(i, j);
        });
      });
    }

    if (state.step === "mlp") {
      document.getElementById("tokenSelect").addEventListener("change", event => {
        state.token = Number(event.target.value);
        render();
      });
    }
  }

  el.nav.addEventListener("click", event => {
    const button = event.target.closest("button[data-step]");
    if (!button) return;
    state.step = button.dataset.step;
    el.nav.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === button));
    render();
  });

  el.layer.addEventListener("change", event => {
    state.layer = Number(event.target.value);
    populateHeads();
    render();
  });

  el.head.addEventListener("change", event => {
    state.head = Number(event.target.value);
    render();
  });

  el.file.addEventListener("change", async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setTrace(JSON.parse(await file.text()));
    } catch (error) {
      alert("Could not load trace: " + error.message);
    }
  });

  if (window.TRACE_DATA) {
    setTrace(window.TRACE_DATA);
  }
})();
