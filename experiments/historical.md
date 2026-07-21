# Archived directions

| Direction | Status | Decision |
|---|---|---|
| Particle candidates | Archived | Sequential baseline; does not fix top-K proposal support. |
| Factor-tree/exhaustive sampling | Archived | Useful historical reference, too costly for active search. |
| Second-order aggregation | Archived | Correct research path, not raw-zone speed. |
| Hierarchical travel kernel | Archived | Compression result, not a standalone runtime solution. |
| Exact heap top-K | Retained oracle | Keep for small-case validation only. |
| Single-suffix factor map | Rejected | Collapses proposals onto one speculative continuation. |
| Eight-suffix factor map | Rejected | Splits a fixed candidate budget too thinly. |
| Factor map depth 6 | Rejected | Long-chain guidance is too weak; weighted Mass@10 0.706. |
| Pure partial-map bootstrap | Rejected | One completed neighbouring factor collapses the beam: pilot Mass@10 0.491 vs 0.535 for factor-map. |
| Full/local map union | Rejected | More candidates do not provide a valid continuation rank: Mass@10 0.505 and 5.7x slower than factor-map. |
| Root-diverse ping-pong bridge | Rejected | Recovered context 26 (Mass@10 0.733→0.918), but a 15-context stratified audit was identical to baseline at 80% more bounded time (2.84→5.11 ms/context). |

Archived code and retired benchmark harnesses: `research-archive-2026-07-21`.
The retained oracle is `rust/oracle.rs`. Details and compact lessons
are in [`lessons-learned.md`](lessons-learned.md).
