# RealAgainst Peg-Slot Evaluation Results

Evaluation uses [`realagainst_peg_slot_openpi_rlinf_eval.yaml`](realagainst_peg_slot_openpi_rlinf_eval.yaml)
with `RealAgainstPegSlot-v0`, sequential seeds starting at 0, 16 parallel
environments, 6 rollout epochs, and a 100-step episode limit. Success is the
episode-level `success_once` metric over 96 trajectories.

| Policy | Successes | Success rate | Mean episode length | Mean return |
| --- | ---: | ---: | ---: | ---: |
| Pi0.5 SFT, step 30,000 | 26 / 96 | 27.083% | 81.875 | 1.2917 |
| Pi0.5 DAgger, step 10,000 | 69 / 96 | 71.875% | 61.875 | 3.6667 |

The DAgger policy improves success by 44.792 percentage points (43 additional
successful episodes, or 2.65x the SFT success rate).

The DAgger run was initialized from the SFT step-30,000 checkpoint and trained
for 10,000 update steps. It collected 500 expert episodes using the configured
linear schedule: 16 episodes are available initially, the cumulative target
increases linearly, and reaches 500 at update step 5,000.

Local evaluation artifacts:

- SFT: `logs/20260824-17:05:31-realagainst_peg_slot_openpi_rlinf_eval`
- DAgger: `logs/20260826-12:51:47-realagainst_peg_slot_openpi_rlinf_eval`
