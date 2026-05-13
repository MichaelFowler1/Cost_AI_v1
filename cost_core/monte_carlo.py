"""
monte_carlo.py - Monte Carlo simulation engine for the cost-estimation core.
Provides probabilistic cost modeling using various statistical distributions.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SimulationResult:
    """
    Container for the results of a Monte Carlo cost simulation.
    
    Attributes:
        samples: The raw array of simulated total costs.
        mean: The arithmetic mean of the simulated costs.
        p50: The 50th percentile (median) cost.
        p80: The 80th percentile cost (80% confidence level).
        p90: The 90th percentile cost (90% confidence level).
    """
    samples: np.ndarray
    mean: float
    p50: float
    p80: float
    p90: float


def _draw_samples(dist_params: Dict[str, Any], n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """
    Internal helper to draw samples from a specified distribution.
    
    Args:
        dist_params: Dictionary containing 'type' and distribution-specific parameters.
        n_iter: Number of samples to draw.
        rng: NumPy random generator instance.
        
    Returns:
        np.ndarray: Array of sampled values.
        
    Raises:
        ValueError: If the distribution type is unsupported or parameters are missing.
    """
    dist_type = dist_params.get("type", "").lower()
    
    try:
        if dist_type == "normal":
            return rng.normal(loc=dist_params["loc"], scale=dist_params["scale"], size=n_iter)
        
        elif dist_type == "lognormal":
            return rng.lognormal(mean=dist_params["mean"], sigma=dist_params["sigma"], size=n_iter)
            
        elif dist_type == "triangular":
            return rng.triangular(left=dist_params["left"], mode=dist_params["mode"], right=dist_params["right"], size=n_iter)
            
        else:
            raise ValueError(f"Unsupported distribution type: '{dist_type}'. Allowed: normal, lognormal, triangular.")
            
    except KeyError as e:
        raise ValueError(f"Missing required parameter {e} for distribution type '{dist_type}'.")


def run_monte_carlo(
    n_iter: int, 
    unit_cost_dist: Dict[str, Any], 
    quantity_dist: Dict[str, Any],
    seed: int | None = None
) -> SimulationResult:
    """
    Runs a Monte Carlo simulation to calculate total cost distribution.
    Total Cost = Unit Cost * Quantity.
    
    Args:
        n_iter: Number of simulation iterations.
        unit_cost_dist: Dictionary defining the unit cost distribution.
        quantity_dist: Dictionary defining the quantity distribution.
        seed: Optional random seed for deterministic outputs.
        
    Returns:
        SimulationResult: The calculated statistics and raw samples.
    """
    logger.info(f"Starting Monte Carlo simulation with {n_iter} iterations.")
    rng = np.random.default_rng(seed)
    
    # Draw samples
    unit_costs = _draw_samples(unit_cost_dist, n_iter, rng)
    quantities = _draw_samples(quantity_dist, n_iter, rng)
    
    # Ensure no negative costs or quantities, as they are non-physical in this context
    unit_costs = np.maximum(unit_costs, 0.0)
    quantities = np.maximum(quantities, 0.0)
    
    # Calculate Total Cost
    total_costs = unit_costs * quantities
    
    # Compute summary statistics
    mean_cost = float(np.mean(total_costs))
    p50_cost = float(np.percentile(total_costs, 50))
    p80_cost = float(np.percentile(total_costs, 80))
    p90_cost = float(np.percentile(total_costs, 90))
    
    logger.info(f"Simulation complete. Mean: {mean_cost:.2f}, P80: {p80_cost:.2f}")
    
    return SimulationResult(
        samples=total_costs,
        mean=mean_cost,
        p50=p50_cost,
        p80=p80_cost,
        p90=p90_cost
    )


def plot_distribution(result: SimulationResult, bins: int = 50) -> None:
    """
    Plots a histogram of the simulation results with confidence intervals marked.
    
    Args:
        result: The SimulationResult object containing samples and metrics.
        bins: Number of histogram bins.
    """
    logger.info("Generating distribution plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(result.samples, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.7)
    
    # Add vertical lines for percentiles and mean
    ax.axvline(result.mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {result.mean:,.2f}")
    ax.axvline(result.p50, color="orange", linestyle="-", linewidth=2, label=f"P50: {result.p50:,.2f}")
    ax.axvline(result.p80, color="green", linestyle="-.", linewidth=2, label=f"P80: {result.p80:,.2f}")
    ax.axvline(result.p90, color="purple", linestyle=":", linewidth=2, label=f"P90: {result.p90:,.2f}")
    
    # Formatting
    ax.set_title("Monte Carlo Total Cost Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Cost", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    # Use pandas formatting for x-axis if desired, but default matplotlib usually suffices
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Setup basic logging for demo
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define distributions
    # Example: Unit cost is lognormal, Quantity is triangular
    unit_cost_config = {
        "type": "lognormal",
        "mean": np.log(150),  # Underlying normal mean
        "sigma": 0.2          # Underlying normal standard deviation
    }
    
    quantity_config = {
        "type": "triangular",
        "left": 40,
        "mode": 50,
        "right": 75
    }

    # Run Simulation
    sim_result = run_monte_carlo(
        n_iter=10000,
        unit_cost_dist=unit_cost_config,
        quantity_dist=quantity_config,
        seed=42 # Fixed seed for deterministic demo
    )

    # Print Report
    print("\n--- Monte Carlo Simulation Results ---")
    print(f"Iterations : {len(sim_result.samples):,}")
    print(f"Mean Cost  : ${sim_result.mean:,.2f}")
    print(f"P50 (Base) : ${sim_result.p50:,.2f}")
    print(f"P80 (Safe) : ${sim_result.p80:,.2f}")
    print(f"P90 (Cons.) : ${sim_result.p90:,.2f}")
    print("--------------------------------------\n")

    # Plot
    plot_distribution(sim_result)