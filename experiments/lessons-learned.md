# Decisions

- Use bounded bidirectional top-K; validate quality with exact top-K mass.
- Proposal support is the main loss source; widening every list is not an
  efficient fix.
- Refresh adds useful forward support without replacing the reverse/home
  frontier. Seam lookahead had identical mass and nearly doubled runtime, so
  it was removed.
- A full-domain, root-diverse forward bridge can repair an isolated repeated-
  anchor miss, but did not improve the stratified audit and cost 80% more.
- Exact search must prove or fail at its state budget; it is an oracle, never a
  production fallback.
- Scalar bounds, sparse random traversal, particles, factor trees,
  second-order raw-zone recursion, and hierarchical kernels are archived.

Measurements live beside the relevant experiment; source is tagged
`research-archive-2026-07-21`.
