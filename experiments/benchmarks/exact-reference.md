# Exact oracle

`DestinationPlanSearch.exact_top_k()` is the validation authority for small
contexts. It returns proven top-K plans or raises at `max_states`; it never
substitutes an approximation. It is used by the active quality analyses.

The retained heap oracle combines home splitting, relaxed bounds, and lazy
sibling expansion. Its state budget remains essential for loose-bound and
repeated-anchor contexts.

Historical factor-tree sampling and enumeration measurements are preserved by
the `research-archive-2026-07-21` Git tag. Their conclusion remains: contiguous scans and
finite-contribution evaluation beat scalar pruning and sparse random
traversal; exact all-zone work is not the active search path.
