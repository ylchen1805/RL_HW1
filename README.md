# PA1: Multi-Armed Bandit Experiments

This project implements six multi-armed bandit algorithms and compares
their performance in stationary and non-stationary environments.

## 1. Language and dependencies

- Language: Python
- Tested Python version: 3.11.15
- Dependencies: NumPy and Matplotlib
- Environment and dependency management: uv

The exact dependency versions are recorded in `uv.lock`.

Check the Python version:

```bash
uv run python --version
```

## 2. Installation

Install uv if it is not already available.

Open a terminal in the project root, where `pyproject.toml` and
`uv.lock` are located, then install the locked dependencies:

```bash
uv sync --locked
```

All commands below must be executed from the project root.

The commands use `uv run python`; no custom shell aliases are required.

## 3. Project structure

```text
.
├── main.py
├── mab.py
├── experiment_two.py
├── experiment_three.py
├── pyproject.toml
├── uv.lock
├── algo/
│   ├── __init__.py
│   ├── base.py
│   ├── greedy.py
│   ├── epsilon_greedy.py
│   ├── decaying_epsilon_greedy.py
│   ├── optimistic_initial_values.py
│   ├── ucb.py
│   ├── thompson_sampling.py
│   └── discounted_thompson_sampling.py
├── analyze/
│   ├── __init__.py
│   ├── greedy.py
│   ├── optimistic.py
│   ├── fixed_alpha.py
│   ├── discounted_thompson.py
│   ├── epsilon_sweep.py
│   ├── ucb_sweep.py
│   ├── ucb_gap_sweep.py
│   └── ucb_k20.py
├── tests/
└── outputs/
```

## 4. Environment and implementation details

Each arm returns:

- `+1` for a success.
- `-1` for a failure.

For an arm with success probability `p`, the expected reward per pull
is `2p - 1`.

Arm indices start at zero. In the stationary experiments, arm 0 is
the unique best arm.

Time steps start at 1.

Ties between maximum action values or scores are broken randomly.
UCB selects untried arms before calculating its exploration bonus.

### Default algorithms

| Algorithm | Default settings |
|---|---|
| Greedy | Q0=0; sample-average update |
| Epsilon-greedy | epsilon=0.1; Q0=0; sample-average update |
| Decaying epsilon-greedy | epsilon0=1.0; decay=0.99; epsilon_min=0.01; sample-average update |
| Optimistic initial values | Q0=5; fixed alpha=0.1 |
| UCB | c=2*sqrt(2); Q0=0; sample-average update |
| Thompson sampling | Initial Beta(1,1) for each arm |

Thompson sampling uses Beta samples to select arms. Its Q values
are maintained only for statistics, not for action selection.

## 5. Random seeds and reproducibility

The report uses 10 runs and starting seed 0.

For run index `i`, starting from zero:

```text
run_seed = starting_seed + i
```

Each run uses `np.random.SeedSequence(run_seed).spawn(2)` to create
separate random generators for the environment and the algorithm.

Every run creates a new environment and a new algorithm instance.

Parameter comparisons use the same run seeds. This does not imply
identical action or reward trajectories across algorithms.

The analysis scripts for parameter sweeps currently use these
settings directly in the script:

```text
runs = 10
starting_seed = 0
```

Analysis scripts that read existing NPZ files reuse the seeds stored
in those files.

All reported standard deviations are sample standard deviations,
calculated with `ddof=1`.

## 6. Reproduce every figure

Run the commands in the order shown below. Some analysis scripts
depend on NPZ files produced by earlier experiments.

### Experiment 1: Default settings

Settings:

- K=2
- p=[0.6, 0.4]
- T=1000
- 10 runs
- Starting seed=0

#### Figure 1: Six mean cumulative reward curves

```bash
uv run python main.py --exp 1 --runs 10 --seed 0
```

Main outputs:

```text
outputs/exp1/mean_cumulative_reward.png
outputs/exp1/summary.csv
outputs/exp1/<AlgorithmName>.npz
```

#### Greedy: Two individual runs

Run after Experiment 1:

```bash
uv run python -m analyze.greedy
```

This selects the runs with the highest and lowest final cumulative
rewards from the saved Greedy runs.

Output:

```text
outputs/exp1/greedy_selected_runs.png
```

#### Optimistic initial values: Fixed step size versus sample average

Run after Experiment 1:

```bash
uv run python -m analyze.optimistic
```

This compares:

- Greedy with Q0=0 and sample-average updates.
- Optimistic initial values with Q0=5 and alpha=0.1.
- Optimistic initial values with Q0=5 and sample-average updates.

Output:

```text
outputs/exp1/optimistic_update_comparison.png
```

### Experiment 2: Changing the number of arms

```bash
uv run python main.py --exp 2 --runs 10 --seed 0
```

This command runs:

- K=2, 5, 10, 20 with T=1000.
- K=20 with T=10000.

For every setting, arm 0 has success probability 0.6 and all other
arms have success probability 0.4.

Outputs:

```text
outputs/exp2/figure_2a.png
outputs/exp2/figure_2b.png
outputs/exp2/table_2.csv
outputs/exp2/summary.csv
outputs/exp2/K<K>_T<T>/<AlgorithmName>.npz
```

`figure_2a.png` contains four subplots, one for each K.

`figure_2b.png` shows K=20 with T=10000.

### Experiment 3: Switching probabilities

#### Figure 3a: Original algorithms

```bash
uv run python main.py --exp 3 --runs 10 --seed 0
```

Settings:

- K=2
- T=1000
- p=[0.6, 0.4] for steps 1 through 500.
- p=[0.4, 0.6] for steps 501 through 1000.
- 10 runs
- Starting seed=0

The probability switch is implemented in `SwitchingBandit` inside
`experiment_three.py`. The original `mab.py` is unchanged.

The algorithms do not receive the environment probabilities or
switch time.

Outputs:

```text
outputs/exp3/figure_3a.png
outputs/exp3/summary.csv
outputs/exp3/snapshots.csv
outputs/exp3/<AlgorithmName>.npz
```

`snapshots.csv` records Q values and counts after updates at
t=500, 600, 800, and 1000.

#### Figure 3b: Fixed step size and discounted Thompson sampling

First run Experiment 1 and Figure 3a, then execute:

```bash
uv run python -m analyze.fixed_alpha
uv run python -m analyze.discounted_thompson
```

The first command sets alpha=0.1 for all Q-based algorithms.
Optimistic initial values already uses this step size.

The second command adds discounted Thompson sampling with
discount=0.99 and updates Figure 3b. It retains the original
Thompson curve for comparison.

Discounted Thompson sampling reduces the weight of old success
and failure evidence every step while preserving the Beta(1,1)
prior. It does not use the switch time.

Final figure:

```text
outputs/exp3/figure_3b.png
```

Additional outputs include:

```text
outputs/exp3/fixed_alpha_summary.csv
outputs/exp3/FixedAlpha_<AlgorithmName>.npz
outputs/exp3/DiscountedThompsonSampling.npz
```

### Experiment 4a: Epsilon-greedy parameter sweep

```bash
uv run python -m analyze.epsilon_sweep
```

Settings:

- K=2 and K=10
- T=1000
- epsilon in {0, 0.01, 0.05, 0.1, 0.2, 0.5, 1}
- p(arm 0)=0.6; all other arms have p=0.4
- Q0=0
- Sample-average updates
- 10 runs per parameter value
- Starting seed=0

Outputs:

```text
outputs/exp4/figure_4a_K2.png
outputs/exp4/figure_4a_K10.png
outputs/exp4/epsilon_summary.csv
```

### Experiment 4b: UCB parameter sweep

```bash
uv run python -m analyze.ucb_sweep
```

Settings:

- K=2
- p=[0.6, 0.4]
- T=1000 and T=10000
- c in {0, 0.5, 0.75, 1, 1.5, 2, 2*sqrt(2), 4}
- Q0=0
- Sample-average updates
- 10 runs per parameter value
- Starting seed=0

Outputs:

```text
outputs/exp4/figure_4b_T1000.png
outputs/exp4/figure_4b_T10000.png
outputs/exp4/ucb_summary.csv
```

### Experiment 4c: UCB with different probability gaps

```bash
uv run python -m analyze.ucb_gap_sweep
```

Settings:

- K=2
- T=1000
- The same eight c values as Experiment 4b
- p=[0.6, 0.4], [0.8, 0.2], and [0.55, 0.45]
- Q0=0
- Sample-average updates
- 10 runs per parameter value
- Starting seed=0

The reference lines are 0.2t, 0.6t, and 0.1t, respectively.

Outputs:

```text
outputs/exp4/figure_4c_default.png
outputs/exp4/figure_4c_large_gap.png
outputs/exp4/figure_4c_small_gap.png
outputs/exp4/ucb_gap_summary.csv
```

### UCB at K=20 with the selected c

Run Experiment 2 first, then execute:

```bash
uv run python -m analyze.ucb_k20
```

This compares the saved default UCB result against c=1 at:

- K=20
- T=1000
- p(arm 0)=0.6; all other arms have p=0.4
- The same run seeds as Experiment 2

The comparison is printed in the terminal.

Additional output:

```text
outputs/exp4/ucb_K20_c1.npz
```

## 7. Run the smoke tests

```bash
uv run python -m tests.greedy
uv run python -m tests.epsilon_greedy
uv run python -m tests.decaying_epsilon_greedy
uv run python -m tests.optimistic_initial_values
uv run python -m tests.ucb
uv run python -m tests.thompson_sampling
```

The tests check algorithm-specific behavior, legal arm indices,
reward values, counts, updates, and reproducibility.

## 8. Output and interpretation notes

Figures are saved directly to PNG files. No interactive plot
window is required.

The plotted reference line is the expected cumulative reward
from always choosing the best arm. It is not a hard upper bound
on a finite sample of observed rewards.

Repeated commands overwrite files with the same output names.

When reproducing all report figures, use runs=10 and seed=0
consistently. Some analysis scripts compare stored results and
check that their run seeds match.
