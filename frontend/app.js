(() => {
  const state = { trace: null, step: "overview", layer: 0, head: 0, token: 7, attentionCell: null };

  const el = {
    content: document.getElementById("content"),
    empty: document.getElementById("emptyState"),
    summary: document.getElementById("traceSummary"),
    selectors: document.getElementById("selectors"),
    layer: document.getElementById("layerSelect"),
    head: document.getElementById("headSelect"),
    token: document.getElementById("tokenSelect"),
    headLabel: document.getElementById("headLabel"),
    tokenLabel: document.getElementById("tokenLabel"),
    file: document.getElementById("traceFile"),
    nav: document.getElementById("stepNav"),
    status: document.getElementById("dataStatus"),
    dropZone: document.getElementById("dropZone"),
  };

  const esc = (value) => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

  const fmt = (value, digits = 4) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return Number(value).toFixed(digits);
  };

  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
  const flattenFinite = (values) => values.flat(Infinity).map(Number).filter(Number.isFinite);
  const maxAbs = (values) => Math.max(1e-9, ...flattenFinite(values).map(Math.abs));
  const mean = (values) => values.reduce((s, v) => s + Number(v || 0), 0) / Math.max(1, values.length);
  const rms = (values) => Math.sqrt(mean(values.map(v => Number(v || 0) ** 2)));

  function heatStyle(value, scale) {
    if (value === null || value === undefined) return "background:var(--masked)";
    const x = clamp(Number(value) / (scale || 1), -1, 1);
    const alpha = 0.12 + 0.78 * Math.abs(x);
    return x >= 0
      ? `background:rgba(52,93,210,${alpha})`
      : `background:rgba(205,78,92,${alpha})`;
  }

  function probabilityStyle(value, max) {
    const x = clamp(Number(value) / Math.max(max, 1e-12), 0, 1);
    return `background:rgba(52,93,210,${0.08 + 0.88 * x})`;
  }

  function sectionTitle(kicker, title, subtitle) {
    return `<div class="page-head"><div class="eyebrow dark">${kicker}</div><h2>${title}</h2><p>${subtitle}</p></div>`;
  }

  function tokenButton(token, selected = false, extra = "") {
    return `<button class="token-pill ${selected ? "selected" : ""}" ${extra}>
      <strong>${esc(token.text || "␠")}</strong><span>${token.id}</span>
    </button>`;
  }

  function vectorHeat(values, { limit = values.length, labels = false } = {}) {
    const shown = values.slice(0, limit);
    const scale = maxAbs(shown);
    return `<div class="vector-heat ${labels ? "with-labels" : ""}">${shown.map((v, i) =>
      `<span class="vector-square" style="${heatStyle(v, scale)}" title="dim ${i}: ${fmt(v, 6)}">${labels ? i : ""}</span>`
    ).join("")}${values.length > limit ? `<span class="vector-more">+${values.length - limit}</span>` : ""}</div>`;
  }

  function topMagnitude(values, count = 12) {
    return values.map((value, index) => ({ index, value: Number(value) }))
      .filter(x => Number.isFinite(x.value))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, count);
  }

  function magnitudeBars(items, labelPrefix = "dim") {
    const max = Math.max(1e-9, ...items.map(x => Math.abs(x.value)));
    return `<div class="magnitude-list">${items.map(x => `
      <div class="magnitude-row">
        <span>${labelPrefix} ${x.index}</span>
        <div class="mag-track"><div class="mag-bar ${x.value < 0 ? "negative" : ""}" style="width:${Math.abs(x.value) / max * 100}%"></div></div>
        <strong>${fmt(x.value, 3)}</strong>
      </div>`).join("")}</div>`;
  }

  function validateTrace(trace) {
    if (!trace || !Array.isArray(trace.tokens) || !Array.isArray(trace.layers) || !trace.embeddings || !trace.final) {
      throw new Error("This JSON is not an LLM Visualize forward-pass trace.");
    }
    if (!trace.layers.length || !trace.tokens.length) throw new Error("The trace contains no layers or tokens.");
    return trace;
  }

  function populateSelectors() {
    el.layer.innerHTML = state.trace.layers.map((_, i) => `<option value="${i}">Layer ${i + 1}</option>`).join("");
    el.token.innerHTML = state.trace.tokens.map((t, i) => `<option value="${i}">${esc(t.text)} · pos ${i}</option>`).join("");
    state.layer = clamp(state.layer, 0, state.trace.layers.length - 1);
    state.token = clamp(state.token, 0, state.trace.tokens.length - 1);
    el.layer.value = String(state.layer);
    el.token.value = String(state.token);
    populateHeads();
  }

  function populateHeads() {
    const heads = state.trace.layers[state.layer]?.attention?.num_heads || 1;
    state.head = clamp(state.head, 0, heads - 1);
    el.head.innerHTML = Array.from({ length: heads }, (_, i) => `<option value="${i}">Head ${i + 1}</option>`).join("");
    el.head.value = String(state.head);
  }

  function setTrace(trace, source = "trace") {
    state.trace = validateTrace(trace);
    state.layer = 0;
    state.head = 0;
    state.token = Math.max(0, trace.tokens.length - 1);
    state.attentionCell = null;

    const m = trace.metadata || {};
    el.summary.textContent = `${m.model_class || "GPT"} · ${m.num_layers ?? trace.layers.length} layers · ${m.embedding_dim ?? "?"}d · ${trace.tokens.length} tokens`;
    el.status.textContent = `Loaded ${source}`;
    el.status.classList.add("ready");
    el.empty.classList.add("hidden");
    el.content.classList.remove("hidden");
    populateSelectors();
    render();
  }

  function renderOverview() {
    const m = state.trace.metadata || {};
    const prompt = state.trace.tokens.map(t => t.text).join(" ");
    const stages = [
      ["embeddings", "Token + position", "128 numbers per token"],
      ["attention", "Self-attention", "Q/K/V + softmax"],
      ["residual", "Residual stream", "Add information back"],
      ["mlp", "MLP / GELU", "Per-token feature transform"],
      ["logits", "Vocabulary logits", "Next-token probabilities"],
    ];

    return sectionTitle("START HERE", "Follow one real inference", "Nothing here is simulated. Every number comes from the saved forward pass of your trained mini-world model.") + `
      <div class="hero-card">
        <div class="hero-prompt"><span>Prompt</span><strong>${esc(prompt)}</strong></div>
        <div class="hero-stats">
          <div><strong>${m.num_layers ?? state.trace.layers.length}</strong><span>layers</span></div>
          <div><strong>${m.embedding_dim ?? "?"}</strong><span>embedding dims</span></div>
          <div><strong>${state.trace.layers[0]?.attention?.num_heads ?? "?"}</strong><span>heads / layer</span></div>
          <div><strong>${state.trace.tokens.length}</strong><span>tokens</span></div>
        </div>
      </div>

      <div class="flow-list">
        ${stages.map(([step, name, detail], i) => `
          <button class="flow-stage" data-jump="${step}">
            <span class="stage-number">${String(i + 1).padStart(2, "0")}</span>
            <div><strong>${name}</strong><span>${detail}</span></div>
            <span class="stage-arrow">→</span>
          </button>`).join("")}
      </div>

      <div class="explain-card">
        <strong>The mental model</strong>
        <p>The weights are already learned and stay fixed. This debugger follows the temporary activations created when this exact prompt passes through those weights.</p>
      </div>`;
  }

  function renderEmbeddings() {
    const e = state.trace.embeddings;
    const token = state.trace.tokens[state.token];
    const t = e.token[state.token];
    const p = e.position[state.token];
    const c = e.combined[state.token];
    const preview = Math.min(32, t.length);

    const rows = Array.from({ length: preview }, (_, i) => ({ i, token: t[i], position: p[i], combined: c[i] }));
    return sectionTitle("02 · EMBEDDINGS", `What does “${esc(token.text)}” look like as numbers?`, "The model cannot work with words directly. It looks up a learned token vector, adds a learned position vector, and sends the sum into layer 1.") + `
      <div class="token-strip">${state.trace.tokens.map((tok, i) => tokenButton(tok, i === state.token, `data-token="${i}"`)).join("")}</div>

      <div class="equation-card">
        <div><span>token embedding</span>${vectorHeat(t, { limit: 48 })}</div>
        <span class="math-symbol">+</span>
        <div><span>position ${state.token}</span>${vectorHeat(p, { limit: 48 })}</div>
        <span class="math-symbol">=</span>
        <div><span>combined input</span>${vectorHeat(c, { limit: 48 })}</div>
      </div>

      <div class="grid two">
        <div class="card">
          <h3>First ${preview} dimensions</h3>
          <div class="dim-table-wrap"><table class="dim-table">
            <thead><tr><th>dim</th><th>token</th><th>position</th><th>sum</th></tr></thead>
            <tbody>${rows.map(r => `<tr><td>${r.i}</td><td>${fmt(r.token,3)}</td><td>${fmt(r.position,3)}</td><td><strong>${fmt(r.combined,3)}</strong></td></tr>`).join("")}</tbody>
          </table></div>
        </div>
        <div class="card">
          <h3>What changed?</h3>
          <div class="metric-stack">
            <div><span>Token vector RMS</span><strong>${fmt(rms(t),3)}</strong></div>
            <div><span>Position vector RMS</span><strong>${fmt(rms(p),3)}</strong></div>
            <div><span>Combined vector RMS</span><strong>${fmt(rms(c),3)}</strong></div>
          </div>
          <p class="muted-text">The same word can appear twice, but its position vector is different. That makes the two “cat” occurrences start with different combined representations.</p>
        </div>
      </div>`;
  }

  function attentionMath(i, j) {
    const a = state.trace.layers[state.layer].attention;
    const q = a.q[state.head][i];
    const k = a.k[state.head][j];
    const products = q.map((value, d) => Number(value) * Number(k[d]));
    const dot = products.reduce((s, v) => s + v, 0);
    const scaled = dot / Math.sqrt(a.head_dim);
    const weight = a.weights[state.head][i][j];
    const top = products.map((value, index) => ({ index, value })).sort((x, y) => Math.abs(y.value) - Math.abs(x.value)).slice(0, 8);
    const qi = state.trace.tokens[i];
    const kj = state.trace.tokens[j];

    return `<div class="calculation">
      <div class="calc-title"><span>Selected connection</span><strong>${esc(qi.text)} → ${esc(kj.text)}</strong></div>
      <div class="calc-steps">
        <div><span>1</span><p>Take Q for <strong>${esc(qi.text)}</strong> and K for <strong>${esc(kj.text)}</strong>.</p></div>
        <div><span>2</span><p>Dot product: <code>Q · K = ${fmt(dot, 5)}</code></p></div>
        <div><span>3</span><p>Scale by √${a.head_dim}: <code>${fmt(dot,5)} / √${a.head_dim} = ${fmt(scaled,5)}</code></p></div>
        <div><span>4</span><p>Softmax across the allowed keys gives <strong>${(Number(weight) * 100).toFixed(2)}%</strong> attention.</p></div>
      </div>
      <div class="calc-contrib"><span>Largest Q×K dimension contributions</span>${magnitudeBars(top, "dim")}</div>
    </div>`;
  }

  function renderAttention() {
    const a = state.trace.layers[state.layer].attention;
    const weights = a.weights[state.head];
    const tokens = state.trace.tokens;
    const max = Math.max(1e-9, ...flattenFinite(weights));
    const selected = state.attentionCell || { i: tokens.length - 1, j: Math.max(0, tokens.length - 2) };
    const i = clamp(selected.i, 0, tokens.length - 1);
    const j = clamp(selected.j, 0, tokens.length - 1);
    state.attentionCell = { i, j };

    return sectionTitle("03 · SELF-ATTENTION", `Layer ${state.layer + 1}, head ${state.head + 1}`, "Rows ask “what should this token look at?” Columns are the earlier/current tokens it is allowed to use. Click any cell to inspect the real Q·K calculation.") + `
      <div class="attention-layout">
        <div class="card attention-card">
          <div class="matrix-legend"><span>weak</span><div></div><span>strong</span></div>
          <div class="matrix-wrap"><table class="attention-matrix">
            <thead><tr><th>Q ↓ / K →</th>${tokens.map(t => `<th>${esc(t.text)}</th>`).join("")}</tr></thead>
            <tbody>${weights.map((row, rowIndex) => `<tr><th>${esc(tokens[rowIndex].text)}</th>${row.map((v, colIndex) => {
              const masked = a.scaled_scores[state.head][rowIndex][colIndex] === null;
              if (masked) return `<td><button disabled class="masked">×</button></td>`;
              const active = rowIndex === i && colIndex === j;
              return `<td><button class="att-cell ${active ? "selected" : ""}" data-i="${rowIndex}" data-j="${colIndex}" style="${probabilityStyle(v,max)}">${(Number(v)*100).toFixed(0)}%</button></td>`;
            }).join("")}</tr>`).join("")}</tbody>
          </table></div>
        </div>
        <div class="card detail-card" id="attentionDetail">${attentionMath(i, j)}</div>
      </div>

      <div class="mini-note"><strong>Important:</strong> a high attention weight does not by itself mean “this token is important.” It means this head routes more of that token’s value vector into the current token at this layer.</div>`;
  }

  function renderResidual() {
    const layer = state.trace.layers[state.layer];
    const before = state.layer === 0 ? state.trace.embeddings.combined[state.token] : state.trace.layers[state.layer - 1].output[state.token];
    const afterAttention = layer.after_attention_residual[state.token];
    const afterMlp = layer.output[state.token];
    const attDelta = afterAttention.map((v, i) => Number(v) - Number(before[i]));
    const mlpDelta = afterMlp.map((v, i) => Number(v) - Number(afterAttention[i]));
    const token = state.trace.tokens[state.token];

    return sectionTitle("04 · RESIDUAL STREAM", `How layer ${state.layer + 1} changes “${esc(token.text)}”`, "A Transformer block does not replace the representation. Attention and the MLP each write a change into the existing residual stream.") + `
      <div class="token-strip">${state.trace.tokens.map((tok, idx) => tokenButton(tok, idx === state.token, `data-token="${idx}"`)).join("")}</div>
      <div class="residual-flow">
        <div class="residual-node"><span>before layer</span><strong>RMS ${fmt(rms(before),3)}</strong>${vectorHeat(before,{limit:48})}</div>
        <div class="residual-plus">+<small>attention writes</small></div>
        <div class="residual-node delta"><span>attention Δ</span><strong>RMS ${fmt(rms(attDelta),3)}</strong>${vectorHeat(attDelta,{limit:48})}</div>
        <div class="residual-equals">=</div>
        <div class="residual-node"><span>after attention</span><strong>RMS ${fmt(rms(afterAttention),3)}</strong>${vectorHeat(afterAttention,{limit:48})}</div>
      </div>
      <div class="residual-flow second">
        <div class="residual-node"><span>after attention</span><strong>RMS ${fmt(rms(afterAttention),3)}</strong>${vectorHeat(afterAttention,{limit:48})}</div>
        <div class="residual-plus">+<small>MLP writes</small></div>
        <div class="residual-node delta"><span>MLP Δ</span><strong>RMS ${fmt(rms(mlpDelta),3)}</strong>${vectorHeat(mlpDelta,{limit:48})}</div>
        <div class="residual-equals">=</div>
        <div class="residual-node"><span>layer output</span><strong>RMS ${fmt(rms(afterMlp),3)}</strong>${vectorHeat(afterMlp,{limit:48})}</div>
      </div>`;
  }

  function renderMlp() {
    const mlp = state.trace.layers[state.layer].mlp;
    const token = state.trace.tokens[state.token];
    const before = mlp.w1_output[state.token];
    const after = mlp.gelu_output[state.token];
    const strongest = topMagnitude(after, 16);
    const suppressed = before.map((v, i) => ({ index: i, before: Number(v), after: Number(after[i]), change: Number(after[i]) - Number(v) }))
      .sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, 10);

    return sectionTitle("05 · MLP / GELU", `Feature activations for “${esc(token.text)}”`, "After attention, each token independently passes through W1 → GELU → W2. GELU changes which hidden features are allowed to pass strongly.") + `
      <div class="token-strip">${state.trace.tokens.map((tok, idx) => tokenButton(tok, idx === state.token, `data-token="${idx}"`)).join("")}</div>
      <div class="grid two">
        <div class="card">
          <h3>Strongest GELU activations</h3>
          ${magnitudeBars(strongest, "neuron")}
          <p class="muted-text">Hidden width: ${mlp.hidden_dim}. The debugger ranks activations by absolute magnitude for this token.</p>
        </div>
        <div class="card">
          <h3>What GELU changed most</h3>
          <div class="change-table">${suppressed.map(x => `<div><span>#${x.index}</span><code>${fmt(x.before,3)}</code><b>→</b><code>${fmt(x.after,3)}</code></div>`).join("")}</div>
          <p class="muted-text">Negative values are often pushed closer to zero; positive values mostly pass through, but GELU is smooth rather than a hard cutoff.</p>
        </div>
      </div>
      <div class="card w2-card"><h3>W2 writes the hidden features back into the ${state.trace.metadata.embedding_dim}-dimensional residual stream</h3>${vectorHeat(mlp.w2_output[state.token], {limit:64})}</div>`;
  }

  function renderLogits() {
    const preds = state.trace.final.last_token_predictions || [];
    const prompt = state.trace.tokens.map(t => t.text).join(" ");
    const max = Math.max(1e-12, ...preds.map(p => Number(p.probability)));
    const top = preds[0];

    return sectionTitle("06 · NEXT TOKEN", `After “${esc(prompt)}”`, "The final token representation is projected to one logit per vocabulary token. Softmax turns those logits into probabilities.") + `
      <div class="prediction-hero">
        <span>Model's top next token</span>
        <strong>${esc(top?.token ?? "?")}</strong>
        <b>${top ? (Number(top.probability) * 100).toFixed(2) : "0"}%</b>
      </div>
      <div class="card prediction-card">
        ${preds.map(p => `<div class="prediction-row">
          <span class="rank">#${p.rank}</span>
          <strong>${esc(p.token)}</strong>
          <div class="prob-track"><div style="width:${Number(p.probability)/max*100}%"></div></div>
          <span>${(Number(p.probability)*100).toFixed(3)}%</span>
          <code>logit ${fmt(p.logit,3)}</code>
        </div>`).join("")}
      </div>
      <div class="mini-note">This trace predicts only the <strong>next</strong> token after the prompt. Generating a sentence means repeating the whole forward-pass process again after appending each chosen token.</div>`;
  }

  function updateSelectorVisibility() {
    const hasLayer = ["attention", "residual", "mlp"].includes(state.step);
    const hasHead = state.step === "attention";
    const hasToken = ["embeddings", "residual", "mlp"].includes(state.step);
    el.selectors.classList.toggle("hidden", !(hasLayer || hasHead || hasToken));
    el.layer.parentElement.classList.toggle("hidden", !hasLayer);
    el.headLabel.classList.toggle("hidden", !hasHead);
    el.tokenLabel.classList.toggle("hidden", !hasToken);
  }

  function render() {
    if (!state.trace) return;
    updateSelectorVisibility();
    const views = { overview: renderOverview, embeddings: renderEmbeddings, attention: renderAttention, residual: renderResidual, mlp: renderMlp, logits: renderLogits };
    el.content.innerHTML = views[state.step]();
    bindContentActions();
  }

  function navigate(step) {
    if (!step) return;
    state.step = step;
    el.nav.querySelectorAll("button[data-step]").forEach(b => b.classList.toggle("active", b.dataset.step === step));
    render();
  }

  function bindContentActions() {
    el.content.querySelectorAll("[data-jump]").forEach(button => button.addEventListener("click", () => navigate(button.dataset.jump)));
    el.content.querySelectorAll("[data-token]").forEach(button => button.addEventListener("click", () => {
      state.token = Number(button.dataset.token);
      el.token.value = String(state.token);
      render();
    }));
    el.content.querySelectorAll(".att-cell").forEach(button => button.addEventListener("click", () => {
      state.attentionCell = { i: Number(button.dataset.i), j: Number(button.dataset.j) };
      render();
    }));
  }

  async function loadFile(file) {
    if (!file) return;
    try {
      setTrace(JSON.parse(await file.text()), file.name);
    } catch (error) {
      el.status.textContent = `Could not load trace: ${error.message}`;
      el.status.classList.remove("ready");
    }
  }

  async function tryDefaultTrace() {
    if (window.TRACE_DATA) {
      setTrace(window.TRACE_DATA, "embedded trace");
      return;
    }
    try {
      const response = await fetch("mini-world-trace.json", { cache: "no-store" });
      if (!response.ok) throw new Error(String(response.status));
      setTrace(await response.json(), "mini-world-trace.json");
    } catch (_) {
      el.status.textContent = "Trace not bundled — drop or load mini-world-trace.json";
    }
  }

  el.nav.addEventListener("click", event => {
    const button = event.target.closest("button[data-step]");
    if (button) navigate(button.dataset.step);
  });
  el.layer.addEventListener("change", event => {
    state.layer = Number(event.target.value);
    state.attentionCell = null;
    populateHeads();
    render();
  });
  el.head.addEventListener("change", event => { state.head = Number(event.target.value); state.attentionCell = null; render(); });
  el.token.addEventListener("change", event => { state.token = Number(event.target.value); render(); });
  el.file.addEventListener("change", event => loadFile(event.target.files?.[0]));

  ["dragenter", "dragover"].forEach(type => el.dropZone.addEventListener(type, event => {
    event.preventDefault();
    el.dropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach(type => el.dropZone.addEventListener(type, event => {
    event.preventDefault();
    el.dropZone.classList.remove("dragging");
  }));
  el.dropZone.addEventListener("drop", event => loadFile(event.dataTransfer?.files?.[0]));

  tryDefaultTrace();
})();
