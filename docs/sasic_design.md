# **SASIC: Sparsity-Aware Synaptic Input Consolidation**

---

# **1. Motivation and Goals**

Baseline Synaptic Input Consolidation (SIC) performs per-neuron clustering by sweeping over k and running Jenks clustering on the full input weight vector. This produces accurate consolidations but is expensive because:
- Many input taps are rarely active and thus contribute little to model behavior.
- Every neuron explores multiple k values even when a smaller, more targeted search would suffice.

**Sparsity-Aware SIC (SASIC)** introduces **training-free, activation-informed heuristics** that:
- Reduce preprocessing time by focusing clustering effort on meaningful inputs,
- Reduce the number of k evaluations required,
- Preserve accuracy and compression ratios,
- Leave acceptance rules, rewiring semantics, and PoS materialization unchanged,
- Maintain exact baseline behavior when disabled.

SASIC is a drop-in, optional extension to SIC.

---

# **2. Baseline SIC Summary (Reference Behavior)**

For each neuron:
1. Extract input weight vector.
2. For each k from 1 to `sic.max_k_per_neuron`:
    - Run Jenks clustering (CPU or GPU).
    - Construct consolidated candidate.
    - Evaluate on acceptance subset (ASC or validation).
3. Accept the first k that satisfies the acceptance criterion.
4. Move to the next neuron.

Hybrid SIC may offload large neurons/layers to GPU, but the logic above defines the authoritative baseline.

SASIC modifies only how candidate clusterings are generated; acceptance and rewiring remain unchanged.

---

# **3. SASIC Concept**

SASIC incorporates **activation statistics** into the clustering process to prioritize or isolate the most informative inputs.

Key ideas:
- Use a **calibration slice** to compute per-tap activation indicators.
- Use these indicators to:
    - Restrict clustering to active inputs (Active-Subset mode),
    - Down-weight quiet inputs (Weighted mode),
    - Combine both (Hybrid mode),
    - Propose a targeted k (k* heuristic) to reduce the k-sweep.

**Properties:**
- Training-free (forward passes only).
- Input-modality agnostic.
- Fully optional and must reproduce baseline SIC when disabled.
- Acceptance rule, PoS construction, rewiring, and evaluation are unchanged.

---

# **4. Activation Statistics**

## **4.1 Calibration Slice**

- Use a small class-balanced subset of the training or validation data.
- Typical size: $~5–10\%$ or a fixed number of batches (configurable).
- Only forward passes; no gradient computation.

## **4.2 Per-Input Statistics**

For each layer, neuron, and input index $(i)$:
- Activity rate $(a_i \in [0,1])$, such as:
    - $(a_i = P(|x_i| > \tau))$, or
    - mean absolute activation $(\mathbb{E}[|x_i|])$.
- Optional sample weight:   $$ 
s_i = a_i^\gamma,\quad \gamma \ge 1  
$$

## **4.3 Storage and Access**

Store activation statistics as:

```
activation_stats[layer_id][neuron_index][input_index] = a_i
```

This structure must be read-only for SIC and SASIC and must allow indexing by layer and tap during clustering.

---

# **5. SASIC Modes**

SASIC modifies only the clustering input, not acceptance or rewiring.

## **5.1 Mode A — Active-Subset SASIC**

1. Define active set:  
$$
A = { i : a_i \ge \tau_{\text{active}} }  
$$
2. Run Jenks clustering on $(w_i, i \in A)$.
3. For each quiet input $(j \notin A)$, attach it to the **nearest cluster center** by weight distance.
4. Construct full consolidated representation.
5. Pass through the standard acceptance logic.

**Motivation:** Reduces dimensionality of the clustering problem while still assigning quiet inputs consistently.

---

## **5.2 Mode B — Weighted SASIC**

1. Compute sample weights $(s_i = a_i^\gamma)$.
2. Run Jenks clustering on all weights using sample weights in the **clustering objective and GVF computation**.
3. Build the consolidated representation.
4. Apply baseline SIC acceptance rule.

**Motivation:** Centers and partitions reflect contribution of frequently active inputs without discarding quiet ones.

---

## **5.3 Mode C — Hybrid SASIC**

A combination of Active-Subset and Weighted SASIC:
1. Build active set (A).
2. Run **weighted Jenks** on active inputs only.
3. Attach quiet inputs using a rule informed by:
    - weight distance, and
    - activation level (bias toward more active clusters).

This design may be refined empirically; the initial implementation follows the steps above.

---

# **6. k* Heuristic**

Instead of sweeping $k = 1 \dots \text{max}_k$, SASIC may propose an initial $k*$ to reduce evaluations.

## **6.1 Deriving k***

Possible signals:
- GVF elbow in sorted $|w|$ or weights weighted by activations,
- Activation-aware separation metrics,
- Variance explained by the top clusters.

## **6.2 Evaluation Pattern**

Evaluate only:  
$$
k \in \{k* - 1,\; k*,\; k* + 1\}  
$$
while clamping to $([1, \text{max}_k])$.

## **6.3 Reliability and Fallback**

- If $k*$ appears unreliable (e.g., poor GVF shape), fall back to a conservative, limited sweep.
- $k*$ is fully optional; disabling it restores the full baseline sweep.

---

# **7. Configuration Additions**

Extend `sic.yaml`:

```yaml
sasic:
  enabled: false
  mode: "none"             # none, active, weighted, hybrid
  kstar_enabled: false
  calibration_fraction: 0.1
  activation_stat: "p_above"
  activation_threshold: 0.01
  weight_exponent: 1.0     # γ in s_i = a_i^γ
```

Defaults must ensure exact baseline SIC behavior.

---

# **8. Integration Requirements**

## **8.1 Where to Integrate**

Modify per-neuron clustering, located in (names from both documents):
- `sic_alg.py`
- `sic_utils.py`

Integration points:
- Inject activation statistics into clustering.
- Apply the appropriate SASIC mode (active, weighted, hybrid).
- Insert optional $k*$ heuristic before launching the $k$ loop.

## **8.2 Where Not to Modify**

- Model construction,
- Data loading/normalization,
- Acceptance rule,
- PoS construction and rewriting,
- ASC/validation logic.

## **8.3 Activation Pass**

- Run calibration once before SIC begins.
- Log calibration overhead.
- Make stats accessible but immutable.

---

# **9. Metrics, Logging, and Evaluation**

Record:
- Calibration time,
- Per-neuron Jenks clustering time,
- Number of $k$ values attempted per neuron,
- Accepted $k$,
- Total SIC preprocessing time,
- Accuracy before and after SIC/SASIC,
- Multiplication counts and memory proxy metrics.

These logs must remain compatible with baseline SIC outputs.

---

# **10. Implementation Constraints**

- SASIC must be opt-in and fully disabled by default.
- No new heavyweight dependencies; rely on existing PyTorch/Numpy stack.
- Code changes must be localized and minimally invasive.
- Must run efficiently on CPU-only and single-GPU setups.
- When disabled, SASIC must reproduce baseline SIC behavior exactly.

---

# **11. Out-of-Scope (Initial Release)**

The following are possible future extensions but **not** part of initial SASIC:
- Neuromorphic readout pipelines for SIC-compressed models,
- Event sparsity analysis,
- Energy estimation or hardware-level modeling.