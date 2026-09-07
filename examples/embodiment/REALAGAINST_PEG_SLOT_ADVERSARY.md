# RealAgainst Peg-Slot SAC adversary

This experiment freezes the DAgger `global_step_6000` OpenPI robot policy and
trains a separate SAC policy to move the slot. The robot owns the first six
public action dimensions; the adversary owns only `[dx, dy, dyaw]`. The SAC
replay buffer contains only the adversary's 17D privileged state, not RGB.

## Reward and horizon

For a normalized adversary action `a_t`, the per-control-step reward is

```text
r_t = -2 I(first robot success)
      +2 I(step 100 and robot never succeeded)
      -0.002 mean(a_t^2)
      -0.005 mean((a_t - a_{t-1})^2)
      -0.05 boundary_violation_t
```

`boundary_violation_t` is the rejected XY displacement divided by 2 mm plus
the rejected yaw displacement divided by 1 degree. It is zero for a legal
action. There is deliberately no lateral-error, depth, or difficulty shaping,
so transient motion cannot be exploited for reward. A failed step-100 timeout
is converted to a true terminal transition; SAC therefore does not bootstrap
after receiving the `+2` payoff.

The task owns its horizon, success history and reward through
`PegSlotEnv.adversary_episode_steps` and `adversary_timeout_bonus`. The timeout
bonus is paid once per episode, including when stepping continues within an
action chunk. Success on the final step takes precedence over timeout failure.
`PegSlotEpisodeBoundary` is registered only for this task, after ManiSkill's
TimeLimit, so a true task termination is not also reported as truncation.
The shared `ManiskillEnv` contains no adversary rules; it preserves task-provided
`info["episode"]` metrics and adds its generic episode statistics.

## Constraints and privileged state

The environment runs at 20 Hz with a 100-step horizon. Each control step is
limited to 2 mm in XY and 1 degree in yaw. Reset-relative cumulative motion is
projected into a 5 cm XY disk and a +/-30 degree yaw interval.

The 17D privileged state contains peg-tip position in the slot frame (3), peg
tilt error (3), relative-yaw sine/cosine (2), insertion depth (1), normalized
reset-relative slot XY (2), reset-relative slot-yaw sine/cosine (2), previous
adversary action (3), and normalized time (1). The actor and each Q head use
two 128-unit hidden layers. Each Q head emits one scalar for the complete
10-step macro-action.

## Configuration and launch

The four-GPU configuration is
`examples/embodiment/config/realagainst_peg_slot_adversary_sac_dagger6k_sparse.yaml`.
It runs 128 training environments (32 per rollout GPU), evaluates 96 episodes
every 50 SAC updates, and saves the small SAC policy every 100 updates.

From the repository root:

```bash
source .venv/bin/activate
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
bash examples/embodiment/run_async.sh \
  realagainst_peg_slot_adversary_sac_dagger6k_sparse
```

Training logs are placed below a timestamped directory under `logs/`. The
reported `success` metric remains robot success; adversary success at the
horizon is logged as `adversary_timeout`. Evaluation videos are written below
`video/eval` in the run directory.
