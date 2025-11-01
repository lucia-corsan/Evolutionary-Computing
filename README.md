# 🚀 Evolutionary-Computing ✨

A compact research toolkit that uses a simple evolutionary optimizer to synthesize
and select features for regression problems. Great for experiments, quick
prototyping, and reproducible comparisons.

> **Highlights:** ✨
>
> - Lightweight, research-friendly implementation
> - Generates new features using arithmetic and simple transforms
> - Evaluates candidates using cross-validated regressors and early stopping

## Table of contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick example](#quick-example)
- [Typical parameters to tune](#typical-parameters-to-tune)
- [Results and analysis](#results-and-analysis)
- [Contributing](#contributing)

## Features

- ✨ Evolutionary (genetic) algorithm for combining and transforming features
- 📊 Evaluates candidate feature sets using multiple regressors and cross-validation
- ⏱️ Early stopping (patience), complexity penalty, and configurable population/genetic operators
- 🧩 Lightweight and easy to adapt for experiments

## Repository layout

- `evopt.py` - Core implementation: class `EvolutionaryOptimizer` (fit + transform).
- `analysis.ipynb` - Notebook with exploratory analysis and examples.
- `data/` - Example datasets (e.g. `diabetes.csv`, `california.csv`).
- `results/` - Output produced by experiments (e.g. `all_results.csv`).
- `requirements.txt` - Minimal Python dependencies

## Requirements

- Python 3.8+ recommended
- See `requirements.txt` for core dependencies (numpy, scikit-learn).

Install dependencies in Windows PowerShell:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick example

Below is a minimal Python example that shows how to use the `EvolutionaryOptimizer`.
This assumes your dataset CSV has a target column named `target`. If your file
uses a different layout adjust the loading code accordingly.

```python
from evopt import EvolutionaryOptimizer
import pandas as pd

# Load data (adjust path / column names as needed)
df = pd.read_csv('data/diabetes.csv')
X = df.drop(columns=['target']).values
y = df['target'].values

# Configure optimizer (shorter runs for testing)
opt = EvolutionaryOptimizer(
    maxtime=600,        # seconds
    pop_size=20,
    crossover_prob=0.7,
    mutation_prob=0.3,
    patience=20,
    random_state=42
)

# Run optimization (this prints progress and final best fitness)
opt.fit(X, y)

# Transform dataset with best discovered features
X_new = opt.transform(X)
print('Original shape:', X.shape)
print('Transformed shape:', X_new.shape)
```

> 💡 Tip: run a short `maxtime` and small `pop_size` for quick experiments. Increase
> time/population for more thorough searches.

- `fit(X, y)` returns the optimizer instance and prints a final summary.
- `transform(X)` applies the best individual found to any dataset with the same
  number of features.
- The implementation in `evopt.py` adapts the number of CV folds and models
  depending on dataset size and may subsample very large datasets.

## Typical parameters to tune

- `maxtime` - total optimization time budget (seconds)
- `pop_size` - population size
- `crossover_prob`, `mutation_prob` - genetic operator probabilities
- `min_genes`, `max_genes` - min/max length of an individual (number of transformations)
- `complexity_penalty` - penalty per gene to discourage overly complex solutions

## Results and analysis

- ✅ Experiment outputs can be written into `results/` (for example `all_results.csv`).
- 📈 Use `analysis.ipynb` to reproduce plots and further inspect the history recorded
  in `EvolutionaryOptimizer.history` after running `fit`.

## Contributing

1. Open an issue describing the feature or bug.
2. Create a branch, make small changes, and open a PR with a clear description.

If you add new dependencies, please update `requirements.txt`.