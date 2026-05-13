"""
learning_curve.py - Mathematical core for Wright's Learning Curve estimation.
Provides utilities for fitting historical data and forecasting future production costs.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LearningCurveModel:
    """
    Representation of a Wright's Learning Curve model.
    
    Attributes:
        slope: The learning slope (e.g., 0.85 for an 85% curve).
        reference_quantity: The quantity at which the reference cost was observed.
        reference_cost: The cost of the unit at the reference quantity.
    """
    slope: float
    reference_quantity: float
    reference_cost: float

    @property
    def learning_exponent(self) -> float:
        """Calculates the 'b' parameter (slope constant)."""
        return np.log2(self.slope)

    def predict_unit_cost(self, quantity: float) -> float:
        """Predicts the cost of a specific unit index using the power law."""
        # Cost(x) = T1 * x^b
        # First, solve for T1 (Theoretical First Unit): T1 = RefCost / RefQty^b
        t1 = self.reference_cost / (self.reference_quantity ** self.learning_exponent)
        return t1 * (quantity ** self.learning_exponent)


def fit_learning_curve(
    df: pd.DataFrame, 
    quantity_col: str = "unit_quantity", 
    cost_col: str = "unit_cost"
) -> LearningCurveModel:
    """
    Fits a learning curve model to historical data using log-log linear regression.
    
    Args:
        df: DataFrame containing historical production data.
        quantity_col: Column name for unit quantities.
        cost_col: Column name for unit costs.
        
    Returns:
        LearningCurveModel: The fitted model parameters.
    """
    # 1. Validation
    if len(df) < 2:
        raise ValueError("At least 2 data points are required to fit a learning curve.")
    
    if (df[quantity_col] <= 0).any() or (df[cost_col] <= 0).any():
        raise ValueError("Quantities and costs must be positive, non-zero values.")

    logger.info(f"Fitting learning curve on {len(df)} data points.")

    # 2. Linear Regression in Log-Log space
    # log(y) = log(a) + b * log(x)
    log_x = np.log2(df[quantity_col].values)
    log_y = np.log2(df[cost_col].values)
    
    slope_b, intercept_log_a, _, _, _ = stats.linregress(log_x, log_y)
    
    # 3. Derive model parameters
    # slope = 2^b
    learning_slope = 2**slope_b
    
    # Use the mean of the data as the reference point for the model
    ref_qty = df[quantity_col].mean()
    ref_cost = (2**intercept_log_a) * (ref_qty ** slope_b)

    model = LearningCurveModel(
        slope=float(learning_slope),
        reference_quantity=float(ref_qty),
        reference_cost=float(ref_cost)
    )
    
    logger.info(f"Model fit complete: Slope={model.slope:.2%}")
    return model


def forecast_costs(
    model: LearningCurveModel, 
    quantities: list[float] | np.ndarray
) -> pd.DataFrame:
    """
    Generates a cost forecast for a range of quantities.
    
    Returns:
        pd.DataFrame: Contains 'quantity', 'unit_cost', and 'total_cost'.
    """
    q_array = np.sort(np.array(quantities))
    
    unit_costs = np.array([model.predict_unit_cost(q) for q in q_array])
    total_costs = q_array * unit_costs
    
    return pd.DataFrame({
        "quantity": q_array,
        "unit_cost": unit_costs,
        "total_cost": total_costs
    })


if __name__ == "__main__":
    # Setup basic logging for demo
    logging.basicConfig(level=logging.INFO)

    # Demo Data: 85% Learning Curve (Theoretical)
    data = pd.DataFrame({
        "unit_quantity": [1, 2, 4, 8, 16],
        "unit_cost": [100.0, 85.0, 72.25, 61.41, 52.20]
    })

    print("--- Historical Data ---")
    print(data)

    # Fit Model
    lc_model = fit_learning_curve(data)
    print(f"\nFitted Slope: {lc_model.slope:.4f}")

    # Forecast
    future_units = np.array([32, 64, 128])
    forecast_df = forecast_costs(lc_model, future_units)

    print("\n--- Forecasted Data ---")
    print(forecast_df.to_string(index=False))