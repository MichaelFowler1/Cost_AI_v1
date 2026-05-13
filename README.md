# Cost-Estimation Core (cost_core)

This repository contains the foundational Python library and command-line interface (CLI) for our cost-estimation and risk analysis workflows.

The core is designed to help analysts fit log-log learning curves to historical production data, project future unit costs, and run Monte Carlo simulations to understand cost risk and confidence intervals (e.g., P50, P80).

## Features

* **Data Ingestion & Fitting:** Safely load historical cost data and calculate theoretical learning curve slopes (log-log regression).
* **Forecasting:** Project unit and total costs for future production lots based on fitted models.
* **Risk Simulation:** Run custom Monte Carlo simulations using standard statistical distributions to generate defensible risk thresholds.

## Installation

This project requires Python 3.11 or higher. It is recommended to install the library within a virtual environment.

1. Create and activate a virtual environment:

```text
python -m venv .venv
.venv\Scripts\activate

```

2. Install the package in editable mode. This will automatically pull in required dependencies like pandas, numpy, and scipy:

```text
pip install -e .

```

Once installed, the CLI tool `ce-core` will be globally available within your virtual environment.

## Usage Guide

The library exposes a single terminal command `ce-core` with three primary subcommands. You can view the help menu at any time by running `ce-core --help`.

### 1. Fit a Learning Curve

Calculate the learning curve slope from a CSV of historical data. The resulting model parameters are saved as a JSON file.

```text
ce-core fit-curve --csv data.csv --out model_params.json

```

### 2. Forecast Future Costs

Use a fitted model to project the costs for upcoming production lots. Pass the targeted unit quantities as a comma-separated list.

```text
ce-core forecast --model model_params.json --quantities "32,64,128" --out forecast.csv

```

### 3. Run a Monte Carlo Simulation

Run a probabilistic simulation to find the P50, P80, etc. You can pass JSON strings to define the distributions for unit cost and quantity.

*Note: If using PowerShell, wrap the JSON arguments in single quotes to prevent parsing errors.*

```text
ce-core simulate --n-iter 10000 --unit-cost-dist '{"type": "lognormal", "mean": 5.0, "sigma": 0.2}' --quantity-dist '{"type": "triangular", "left": 40, "mode": 50, "right": 75}' --out sim_results.csv

```

## Project Structure

* `/cost_core`: The core mathematical Python modules (data_io, learning_curve, monte_carlo).
* `/cli`: The command-line interface wrappers that expose the core to the terminal.
* `pyproject.toml`: The modern build system configuration defining dependencies and CLI entrypoints.