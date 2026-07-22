# Archived second-order recursion

The rigidity-aware exact recursion was correct and its forward/backward
feasibility corridor improved runtime by about 4.3x on aggregated zones.
At raw 1,110-zone resolution it still took 177.46 s for 1,000 contexts, so it
is not a production replacement for bounded top-K.

Decision: archived in Git tag `research-archive-2026-07-21`. Reopen only for a new
reusable-work hypothesis, not as an all-zone fallback.
