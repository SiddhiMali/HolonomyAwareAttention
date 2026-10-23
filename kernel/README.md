# Triton kernel for holonomy-aware attention

A fused Triton forward kernel for `PathOrderedGeometricMHA`, the path-ordered
geometric attention mechanism in this repo. It is a drop-in replacement for the
module, is trainable end to end, and is verified against the reference
implementations to floating-point precision.

## What the kernel does

The mechanism scores each query against a key that has been parallel-transported
into the query's frame. For positions `i` and `j`, with prefix transports `P_t`
(products of per-edge SO(d) rotations) and loop holonomy `H`:

```
T_ij = P_i P_j^T        (i >= j)
T_ij = P_i H P_j^T      (i <  j, wrap-around)
```

A naive implementation materializes `T_ij` for every pair, an O(L²·d²) tensor.
The kernel never builds it. Because `q_i · (P_i P_j^T k_j) = (P_i^T q_i) · (P_j^T k_j)`,
each rotation is applied once per token. The query-side rotation `P_i` does not
depend on `j`, so it factors out of the value sum and is applied once, after
aggregation:

```
out_i = P_i · Σ_j softmax_j(s_ij) · (P_j^T v_j)
```

Inside the kernel this runs as a FlashAttention-style pass. Each program owns a
tile of queries and streams key/value tiles through an online softmax
(running max, running sum, rescaling). The wrap branch applies `H` to the
transformed keys and values for `i < j` pairs.

Training uses a `torch.autograd.Function`. The forward pass runs the Triton
kernel. The backward pass recomputes the same math in plain PyTorch and lets
autograd differentiate it, so gradients are exact but the backward is not fused.

## Files

| File | Purpose |
|---|---|
| `attention_kernel.py` | The kernel, its launcher, the autograd wrapper, and the drop-in module. Run it directly for the self-test. |
| `attention_fast.py` | Vectorized full-grid implementation. Exact in all modes and masks. Used as the fallback. |
| `attention_fast_chunked.py` | Memory-bounded chunked implementation. Supplies the geometry code and serves as the reference in the eval suite. Supports causal mode with masks that block all `i < j` pairs only, and raises otherwise. |
| `eval_checkpoint_kernel.py` | Four-part evaluation: correctness, speed, memory, task parity. |
| `profiling/` | Scripts that produced the profiling numbers below. |
| `results/` | Raw outputs of every run reported here. |

Expected layout: `utils.py` and `attention.py` sit at the repo root, one level
above this folder. Each script adds the root to `sys.path`.

## Usage

```python
from kernel.attention_kernel import PathOrderedGeometricMHA   # or run from inside kernel/

attn = PathOrderedGeometricMHA(d_model=64, n_heads=4, mode="causal", edge_scale=1.5).cuda()
mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device="cuda"))
y, _ = attn(x, attn_mask=mask)          # x: (B, L, d_model)
```

The interface and parameter names match the original `attention.py`, so trained
checkpoints load unchanged.

### When the kernel runs

The Triton path runs only when **all** of these hold:

- `mode == "causal"`
- `attn_mask` is exactly `tril(L)`. The kernel applies causal masking
  internally and does not read the mask, so any other mask falls back.
- dropout is 0, or the module is in eval mode. The kernel does not apply dropout.
- the input is on CUDA
- `return_geometry=False`

Everything else falls back to `attention_fast`, which is exact for every
mode and mask and returns the full geometry dict. The fallback materializes the
pair grid, so at long sequence lengths it uses much more memory than the kernel.

## Verification

Run the self-test first:

```
python kernel/attention_kernel.py
```

It checks the kernel path against both the chunked reference and the original
`attention.py`, checks every fallback path against `attention.py`, gradchecks the
backward math in float64, and confirms end-to-end gradients are finite.

## Results

Raw output for every number below is in `results/`.
GPU: **<fill in>**

### 1. Correctness: kernel vs reference

| L | max \|Δ output\| | max \|Δ gradient\| |
|---|---|---|
| 8 | 1.19e-07 | 4.92e-07 |
| 16 | 1.19e-07 | 9.54e-07 |
| 32 | 1.19e-07 | 1.28e-06 |
| 64 | 1.04e-07 | 1.64e-06 |

Outputs and gradients both match at float32 precision.

### 2. Speed (batch 8, d_model 64, 4 heads)

| L | forward speedup | forward+backward speedup |
|---|---|---|
| 32 | 1.08× | 0.72× |
| 64 | 1.22× | 0.97× |
| 128 | 0.93× | 1.26× |
| 256 | 1.07× | 1.20× |
| 512 | 0.92× | 1.03× |

The forward is within about ±20% of the reference, with no consistent trend in L.
Forward+backward is sometimes slower, because the backward recomputes the
forward in unfused PyTorch.

### 3. Peak memory during forward

Identical to the reference at every tested length (1.00×, L = 32 to 512).

### 4. Training parity (checkpoint length-extrapolation task, 30 epochs)

| eval length | kernel | reference | Δ |
|---|---|---|---|
| 16 (train) | 0.800 | 0.805 | 0.005 |
| 32 | 0.771 | 0.782 | 0.011 |
| 64 | 0.768 | 0.771 | 0.003 |

A model trained with the kernel reaches the same accuracy as one trained with
the reference, within training noise (max Δ = 0.011).

## Why the speedup is small: profiling

At these sequence lengths the attention step is not the bottleneck. Building the
geometry (`P` and `H`) before attention dominates the runtime.

`profiling/time_holo.py` (B = 128): geometry accounts for roughly 75–100% of the
forward time at L = 7 to 64. The attention share is small enough that it is
within timing noise at some lengths.

`profiling/geometry_split.py` breaks the geometry into its parts:

| L | edge MLP | skew build + matrix_exp | prefix scan |
|---|---|---|---|
| 16 | 2% | 86% | 12% |
| 32 | 2% | 73% | 25% |
| 64 | 1% | 71% | 28% |
| 128 | 1% | 54% | 45% |

`profiling/exp_compare.py` tested alternatives to `torch.matrix_exp` at d = 16,
edge_scale = 1.5 (‖A‖ ≈ 7): truncated Taylor series, scaling-and-squaring, and
eigendecomposition. None was both accurate (error < 1e-5) and faster. The only
accurate alternative, eigendecomposition, ran about 2× slower.

Open question: in `geometry_split`, the "skew build + matrix_exp" step costs a flat
~10–12 ms regardless of how many matrices it processes, while `matrix_exp` alone
took 1.2 ms for 1,024 matrices in `exp_compare`. The likely cause is the Python
double loop in `utils.skew_from_vec`, which issues about 240 small GPU operations
per call at d = 16. `profiling/skew_test.py` tests this and a vectorized
replacement; results pending.

## Related: the vectorized implementation

`attention_fast.py` replaced the original O(L³) pairwise ring walk with prefix
products, reducing the forward to O(L²). It matches `attention.py` to ~1e-7 in
causal and inert modes, with dense and ring masks
(`results/profile_attention.txt`). Measured forward speedup over the original at
L = 64: 81.6× in one run and 104.5× in another (batch 4, d_model 64, 8 heads,
causal).

## Limitations

- The backward pass is recompute-based. A fused backward kernel (key-parallel
  pass for `grad_k`/`grad_v`, query-parallel pass for `grad_q`, `grad_P`
  accumulated in both) is not implemented.
- Tile sizes are fixed at 16 × 16 and untuned.
- Triton's `tl.arange` requires power-of-two sizes, so the per-head dimension
  `d` must be a power of two.
- Tested in float32 only.
- Memory savings over the reference were not observed at L ≤ 512; the
  advantage of never materializing the pair grid is expected only at longer
  sequence lengths.

## Reproduce

```
python kernel/attention_kernel.py                     # self-test
python kernel/eval_checkpoint_kernel.py --correctness --speed --memory
python kernel/eval_checkpoint_kernel.py --task --epochs 30
python kernel/profiling/time_holo.py
python kernel/profiling/geometry_split.py
python kernel/profiling/exp_compare.py
python kernel/profiling/skew_test.py
python kernel/profiling/profile_attention.py
```

The `--task` section also needs the checkpoint-task files
(`qgate_comparev5.py`, `qgate_checkpoint.py`, `qgate_data.py`,
`qgate_energy.py`) on the path.
