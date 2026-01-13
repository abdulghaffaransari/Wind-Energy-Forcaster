"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_time_series(
    data: pd.DataFrame,
    columns: list,
    title: str = "Time Series Plot",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot time series data
    
    Args:
        data: DataFrame with time series data
        columns: List of columns to plot
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        if col in data.columns:
            ax.plot(data.index, data[col], label=col, linewidth=1.5)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot predictions against actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Date index for x-axis
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    x_axis = dates if dates is not None else range(len(y_true))
    
    # Time series plot
    ax1.plot(x_axis, y_true, label='Actual', linewidth=2, alpha=0.7)
    ax1.plot(x_axis, y_pred, label='Predicted', linewidth=2, alpha=0.7)
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel("Wind Generation (MW)", fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    ax2.plot(x_axis, residuals, label='Residuals', linewidth=1.5, alpha=0.7, color='red')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Residuals (MW)", fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Plot comprehensive residual analysis
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals over time
    axes[0, 0].plot(residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("Residuals Distribution")
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Predicted vs Residuals
    axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title("Predicted vs Residuals")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
