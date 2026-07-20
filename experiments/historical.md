# Archived directions

| Direction | Status | Decision |
|---|---|---|
| Particle candidates | Archived | Sequential baseline; does not fix top-K proposal support. |
| Factor-tree/exhaustive sampling | Archived | Useful historical reference, too costly for active search. |
| Second-order aggregation | Archived | Correct research path, not raw-zone speed. |
| Hierarchical travel kernel | Archived | Compression result, not a standalone runtime solution. |
| Exact heap top-K | Retained oracle | Keep for small-case validation only. |

Archived code and retired benchmark harnesses: `research-archive-2026-07-21`.
The retained oracle is `rust/oracle.rs`. Details and compact lessons
are in [`lessons-learned.md`](lessons-learned.md).
