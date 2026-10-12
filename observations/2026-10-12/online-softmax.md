# 2026-10-12 — Commit 03: online softmax for one CPU query

Implemented `online_attention(q, k, v, chunk_size)` in `tileforge/online_softmax.py`. Q is one vector of shape `(D,)`; K and V have shape `(N, D)`. This milestone is CPU FP32 forward-only attention, with no masking or dropout. It does not implement the multiple-query tiled attention planned for commit 04.

## Actual runs

Full commands, outputs, chunk traces, environment version, run timestamp (Asia/Kolkata), and exit codes are saved in [online-softmax-runs.txt](online-softmax-runs.txt). All nine commands exited 0 with finite outputs and agreement against both full softmax attention and SDPA at `rtol=1e-5, atol=1e-6`.

Each command uses `uv run --locked python -m tileforge.online_softmax` followed by the options below. Runs used Python 3.12.14, PyTorch 2.10.0+cpu, FP32, and one compute thread.

| Options | What the run exercises | Max error vs full softmax | Max error vs SDPA |
| --- | --- | ---: | ---: |
| `--trace` | N=4, D=4, two chunks | 5.96046448e-8 | 2.98023224e-8 |
| `--seq-len 9 --head-dim 8 --chunk-size 1 --seed 7` | One key/value per chunk | 4.47034836e-8 | 5.96046448e-8 |
| `--seq-len 9 --head-dim 8 --chunk-size 4 --seed 7` | Incomplete final chunk | 4.47034836e-8 | 5.96046448e-8 |
| `--seq-len 9 --head-dim 8 --chunk-size 32 --seed 7` | Chunk larger than the sequence | 5.96046448e-8 | 2.98023224e-8 |
| `--seq-len 1 --chunk-size 2` | Single key; output equals its value | 0 | 0 |
| `--seq-len 17 --chunk-size 4 --scenario uniform --trace` | All scores zero; output matches mean(V) | 2.98023224e-8 | 2.98023224e-8 |
| `--seq-len 17 --chunk-size 4 --scenario rising --trace` | Scores increase from 1000 to 1020 | 1.19209290e-7 | 5.96046448e-8 |
| `--seq-len 17 --chunk-size 4 --scenario negative --trace` | Scores increase from -1000 to -980 | 1.19209290e-7 | 5.96046448e-8 |
| `--seq-len 8192 --head-dim 64 --chunk-size 127 --seed 11` | 65 chunks, including a partial final chunk | 5.96046448e-8 | 4.47034836e-8 |

## What the results show

- The output remains consistent when the same nine-key input is divided into chunks of 1, 4, or 32. Normalization happens across the entire sequence, not independently inside each chunk.
- In the uniform run, the running sum progresses through 4, 8, 12, 16, and 17. The final output matches `mean(V)` with maximum error `2.98023224e-8`.
- In the rising-score run, the running maximum progresses through 1003.75, 1008.75, 1013.75, 1018.75, and 1020. After the first chunk, `alpha` is about `0.006737947` for each five-point increase, then `0.2865048` for the final 1.25-point increase. Both the denominator and numerator are rescaled by this factor before adding new contributions.
- The positive and negative stress runs have identical score differences and produce the same displayed output. Subtracting the running maximum keeps exponentials bounded even when unshifted FP32 exponentials would overflow at positive scores or underflow at negative scores of this magnitude.
- The N=8192 run matches the full references while the online implementation processes at most 127 scores at a time. It retains two scalar statistics and a D-element weighted numerator between chunks. Its scratch space is O(chunk size + D), excluding the resident inputs/output; autograd is disabled to avoid retaining prior chunk intermediates.

These are correctness observations, not measured latency or memory improvements over commit 02. Commit 02 computes all N queries; this milestone computes just one. The CLI deliberately allocates a full N-element reference score vector outside the online implementation to verify the result. No GPU work or unit tests were performed.

## Next

**04 — CPU tiled attention:** extend the recurrence to multiple query rows and Q/K/V tiles while avoiding the full N × N score matrix.
