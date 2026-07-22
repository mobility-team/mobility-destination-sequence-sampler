# Test map

| Test | Invariant exercised |
|---|---|
| `test_two_step_top_k_matches_the_exact_oracle` | two-step direct scan and bounded report accounting |
| `test_exact_oracle_rejects_negative_intermediate_duration` | shared feasibility rejects an invalid intermediate duration |
| `test_bidirectional_top_k_stitches_complete_plan` | exact stitch ordering, repeated anchors, terminal fixed destination, and no-feasible error |
| `test_bidirectional_top_k_supports_variable_anchor` | optional/variable anchor handling with skipped infeasible contexts |
| `test_bidirectional_top_k_matches_exact_when_the_beam_covers_the_toy_domain` | bounded factor-map path agrees with oracle when support covers the domain |
| `test_exact_oracle_fails_explicitly_at_its_state_budget` | oracle proves or fails; it does not approximate |
| `test_search_requires_both_timing_rigidities` | public input contract requires both rigidity columns |

Add a focused unit or integration case whenever a change alters factor
ownership, anchor compatibility, proposal support, or a public input/output
contract. Keep large table construction in `conftest.py` when it is reusable.
