"""
ce_core_cli.py
Terminal entry point for the cost-estimation engine.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Try to hook into the core logic; exit early if the environment is wonky
try:
    from cost_core import data_io, learning_curve, monte_carlo
except ImportError as e:
    sys.exit(f"Error: Project core modules not found ({e}). Are you in the root directory?")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def abort(msg: str):
    """Utility to print an error and kill the process without a traceback."""
    log.error(msg)
    sys.exit(1)

def run_fit(args):
    csv_path, out_path = Path(args.csv), Path(args.out)
    
    if not csv_path.exists():
        abort(f"Input CSV missing: {csv_path}")

    try:
        log.info(f"Fitting model to {csv_path}...")
        df = data_io.load_cost_csv(csv_path)
        model = learning_curve.fit_learning_curve(df, args.quantity_col, args.cost_col)
        
        # Save parameters
        out_path.write_text(json.dumps({
            "slope": model.slope,
            "reference_quantity": model.reference_quantity,
            "reference_cost": model.reference_cost
        }, indent=4))
        log.info(f"Serialized model parameters to {out_path}")
        
    except Exception as e:
        abort(f"Fitting failed: {e}")

def run_forecast(args):
    model_path, out_path = Path(args.model), Path(args.out)

    if not model_path.exists():
        abort(f"Model file not found: {model_path}")

    try:
        # Quick parse of the comma-separated list
        targets = [float(x.strip()) for x in args.quantities.split(",") if x.strip()]

        log.info(f"Loading model: {model_path}")
        params = json.loads(model_path.read_text())
        model = learning_curve.LearningCurveModel(**params)
        
        log.info(f"Projecting costs for {len(targets)} data points...")
        df = learning_curve.forecast_costs(model, targets)
        df.to_csv(out_path, index=False)
        log.info(f"Forecast saved to {out_path}")
        
    except (json.JSONDecodeError, ValueError) as e:
        abort(f"Input validation error: {e}")
    except Exception as e:
        abort(f"Forecast failed: {e}")

def run_simulate(args):
    try:
        log.info(f"Running {args.n_iter} Monte Carlo iterations...")
        res = monte_carlo.run_monte_carlo(
            n_iter=args.n_iter,
            unit_cost_dist=json.loads(args.unit_cost_dist),
            quantity_dist=json.loads(args.quantity_dist)
        )
        
        # Print summary for the user
        print(f"\n--- Simulation Results ---\nMean: ${res.mean:,.2f}\nP80:  ${res.p80:,.2f}\n")
        
        pd.DataFrame({"sim_total_cost": res.samples}).to_csv(args.out, index=False)
        log.info(f"Raw samples exported to {args.out}")
        
    except json.JSONDecodeError:
        abort("Invalid JSON passed to distribution arguments.")
    except Exception as e:
        abort(f"Simulation engine failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="CE Core CLI: Regression & Risk Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Fit Command
    p_fit = sub.add_parser("fit-curve", help="Analyze historical cost trends")
    p_fit.add_argument("--csv", required=True)
    p_fit.add_argument("--out", required=True)
    p_fit.add_argument("--quantity-col", default="unit_quantity")
    p_fit.add_argument("--cost-col", default="unit_cost")

    # Forecast Command
    p_fcst = sub.add_parser("forecast", help="Predict costs for target lots")
    p_fcst.add_argument("--model", required=True)
    p_fcst.add_argument("--out", required=True)
    p_fcst.add_argument("--quantities", required=True, help="e.g. '50,100,500'")

    # Simulate Command
    p_sim = sub.add_parser("simulate", help="Run probabilistic risk models")
    p_sim.add_argument("--out", required=True)
    p_sim.add_argument("--n-iter", type=int, default=10000)
    p_sim.add_argument("--unit-cost-dist", required=True)
    p_sim.add_argument("--quantity-dist", required=True)

    args = parser.parse_args()

    # Dispatcher map: cleaner than if/elif/elif
    handlers = {
        "fit-curve": run_fit,
        "forecast": run_forecast,
        "simulate": run_simulate
    }
    
    handlers[args.cmd](args)

if __name__ == "__main__":
    main()