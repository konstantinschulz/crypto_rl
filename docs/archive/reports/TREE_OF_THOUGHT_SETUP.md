# Tree-of-Thought Experiment Setup

## Goal
Find a profitable RL trading recipe that is **positive on both Val and Test** under a conservative walk-forward setup.

The current near-miss candidate is `wf_midflow_quality_v2` from `run_experiment10.py`:
- long-ish history (`90d`)
- moderate symbol breadth (`5 symbols`)
- walk-forward validation
- positive Val on some seeds
- Test still slightly negative

## Expert roles

### 1. Reward/Churn Expert
Focus: balance the policy so it trades enough to learn, but not so much that fees and churn dominate.

Typical knobs:
- `--trade-action-bonus`
- `--inactivity-penalty`
- `--trade-rate-penalty`
- `--target-trade-rate`
- `--trade-execution-penalty`
- `--turnover-penalty-rate`
- `--max-trades-per-window`
- `--min-hold-steps`

Failure modes:
- zero-trade collapse
- overtrading relapse
- Val improves while Test stays negative

### 2. Data/Regime Expert
Focus: whether the model simply needs more market diversity.

Typical knobs:
- longer history (`180d` or `365d`)
- more symbols
- broader market regimes
- walk-forward folds that preserve chronology

Failure modes:
- the policy works only on one regime
- seed sensitivity remains high

### 3. Model Capacity Expert
Focus: whether the policy network is too small once the objective is sane.

Typical knobs:
- `--model-arch`
- `--n-steps`
- `--n-epochs`
- `--learning-rate`

Failure modes:
- larger models just overfit the wrong reward
- capacity masks reward issues instead of fixing them

### 4. Validation/Robustness Expert
Focus: avoid “one lucky seed” results.

Typical strategy:
- cheap screen on 1 seed
- confirm on 2–3 seeds only for promising variants
- reject variants that collapse into no-trade or churn

Failure modes:
- unstable seed-to-seed behavior
- one seed suggests profit, another breaks the policy

## Priority order agreed by the experts

### Priority 1: Precision nudge around the best near-miss
Start from `wf_midflow_quality_v2` and test one-knob relaxations.

Why first:
- closest candidate to the target
- already positive Val on some runs
- cheapest chance to flip Test positive without changing the full setup

### Priority 2: Regime expansion if the precision sweep fails
If Test stays negative, move to longer time spans and more symbols.

### Priority 3: Capacity scaling only after the reward and data balance look sane
Only then increase model size/complexity.

## First experiment to run

Run a small focused grid on the current 90d/5-symbol walk-forward setup:
- base: `wf_midflow_quality_v2`
- relax 1: `wf_midflow_quality_v2_relax_tradecap`
- relax 2: `wf_midflow_quality_v2_relax_ratepen`

Suggested command pattern:

```bash
./.conda/bin/python run_experiment10.py \
  --days 90 \
  --max-symbols 5 \
  --train-steps 8000 \
  --seeds 11,23 \
  --variants wf_midflow_quality_v2,wf_midflow_quality_v2_relax_tradecap,wf_midflow_quality_v2_relax_ratepen \
  --out artifacts/experiments/results/exp10_tot_precision_grid.json
```

## Decision rules after the first experiment

- If one variant gets `Val > 0` and `Test > 0`, confirm on more seeds.
- If Test improves but remains slightly negative, try one more single-knob relaxation.
- If everything collapses to zero trades, loosen the suppression balance or lower the budget drag.
- If Test worsens but Val stays positive, the strategy is still too seed-specific and needs regime expansion.

## What to record

For each experiment, note:
- seed stability
- trade count
- Val return
- Test return
- whether the policy collapsed to zero-trade or overtrade behavior

## Working hypothesis
The most likely path to Test positivity is still a **small reward/churn correction** around the current near-miss, not a large model jump.

## First experiment result

The focused precision grid completed on `90d / 5 symbols / 8k steps / seeds 11,23`.

### Observed results

- `wf_midflow_quality_v2`: `med_val=0.165%`, `med_test=-0.060%`, `trade_ok=0.00`
- `wf_midflow_quality_v2_relax_ratepen`: `med_val=0.230%`, `med_test=-0.060%`, `trade_ok=0.00`
- `wf_midflow_quality_v2_relax_tradecap`: `med_val=-0.125%`, `med_test=-0.445%`, `trade_ok=0.50`

### Expert readout

- **Reward/Churn Expert**: `relax_tradecap` is too loose for this setup; it creates a churny failure mode on one seed without solving Test.
- **Validation/Robustness Expert**: `relax_ratepen` is effectively a no-op here; it matches the base behavior and does not flip Test positive.
- **Data/Regime Expert**: the signal is still concentrated in the same regime band; a precision tweak did not expand coverage enough.
- **Model Capacity Expert**: there is no evidence yet that a larger network is the fix; the current issue still looks like objective balance and regime coverage.

### Decision

The first precision sweep did **not** produce a positive-Test candidate. The best next move is a single-knob follow-up that preserves the current policy shape while nudging churn pressure:

- **Recommended follow-up**: `wf_midflow_quality_v2_relax_turnover`
- Keep the same `90d / 5 symbols / 8k steps / seeds 11,23` setup
- Confirm whether lowering turnover pressure improves Test without reintroducing overtrading

If that still fails, the next escalation should be **regime expansion** rather than a larger model.


