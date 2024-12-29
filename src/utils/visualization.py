"""
Visualization utilities for quantum-classical simulations.
Implements plotting functions for wavefunctions, potentials, and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, Union, List, Dict
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from ..quantum.potentials import Potential
from ..quantum.wavefunctions import Wavefunction
from ..utils.analysis import StatisticalResult

@dataclass
class PlotStyle:
    """Configuration for plot styling"""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    cmap: str = 'viridis'
    style: str = 'default'
    context: str = 'notebook'

class QuantumVisualizer:
    """Visualization tools for quantum mechanical systems"""
    
    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = style or PlotStyle()
        # Set up plotting style
        plt.style.use(self.style.style)
        sns.set_context(self.style.context)
    
    def plot_wavefunction(self, 
                         wavefunction: Wavefunction,
                         potential: Optional[Potential] = None,
                         show_probability: bool = True,
                         ax: Optional[Axes] = None) -> Axes:
        """
        Plot wavefunction and optionally the potential
        
        Args:
            wavefunction: Quantum wavefunction
            potential: Optional potential energy function
            show_probability: If True, plot |ψ|² instead of Re(ψ)
            ax: Optional matplotlib axes for plotting
            
        Returns:
            matplotlib axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        
        x = wavefunction.x
        
        if show_probability:
            # Plot probability density
            prob = wavefunction.probability_density()
            ax.plot(x, prob, label='|ψ(x)|²', color='blue')
            ax.set_ylabel('Probability Density')
        else:
            # Plot real and imaginary parts
            ax.plot(x, wavefunction.psi.real, label='Re(ψ)', color='blue')
            ax.plot(x, wavefunction.psi.imag, label='Im(ψ)', color='red', linestyle='--')
            ax.set_ylabel('Wavefunction')
        
        if potential is not None:
            # Add potential energy (scaled to fit)
            V = potential(x)
            V_scaled = V / np.max(np.abs(V)) * np.max(np.abs(wavefunction.psi))
            ax.plot(x, V_scaled, label='V(x) (scaled)', color='gray', alpha=0.5)
        
        ax.set_xlabel('Position (x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_density_matrix(self,
                          rho: np.ndarray,
                          x: np.ndarray,
                          ax: Optional[Axes] = None) -> Axes:
        """
        Plot density matrix as a 2D heatmap
        
        Args:
            rho: Density matrix array
            x: Position grid points
            ax: Optional matplotlib axes
            
        Returns:
            matplotlib axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        
        im = ax.imshow(np.abs(rho), cmap=self.style.cmap,
                      extent=[x[0], x[-1], x[0], x[-1]],
                      origin='lower')
        
        ax.set_xlabel('x')
        ax.set_ylabel("x'")
        ax.set_title('Density Matrix |ρ(x,x\')|')  # Fixed escaped quote
        plt.colorbar(im, ax=ax)
        
        return ax

class StatisticalVisualizer:
    """Visualization tools for statistical analysis"""
    
    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = style or PlotStyle()
        plt.style.use(self.style.style)
        sns.set_context(self.style.context)
    
    def plot_correlation_function(self,
                                times: np.ndarray,
                                correlation: np.ndarray,
                                ax: Optional[Axes] = None) -> Axes:
        """
        Plot correlation function
        
        Args:
            times: Time points
            correlation: Correlation function values
            ax: Optional matplotlib axes
            
        Returns:
            matplotlib axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        
        ax.plot(times, correlation)
        ax.axhline(y=np.exp(-1), color='gray', linestyle='--', 
                  label='e⁻¹', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation Function')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_blocking_analysis(self,
                             block_data: Dict[int, float],
                             ax: Optional[Axes] = None) -> Axes:
        """
        Plot blocking analysis results
        
        Args:
            block_data: Dictionary of block sizes and errors
            ax: Optional matplotlib axes
            
        Returns:
            matplotlib axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        
        block_sizes = np.array(list(block_data.keys()))
        errors = np.array(list(block_data.values()))
        
        ax.semilogx(block_sizes, errors, 'o-')
        ax.set_xlabel('Block Size')
        ax.set_ylabel('Statistical Error')
        ax.set_title('Blocking Analysis')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_error_distribution(self,
                              stat_result: StatisticalResult,
                              data: np.ndarray,
                              ax: Optional[Axes] = None) -> Axes:
        """
        Plot error distribution with confidence intervals
        
        Args:
            stat_result: Statistical analysis result
            data: Raw data points
            ax: Optional matplotlib axes
            
        Returns:
            matplotlib axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        
        # Plot histogram of data
        sns.histplot(data, stat='density', ax=ax, alpha=0.5)
        
        # Plot normal distribution fit
        x = np.linspace(data.min(), data.max(), 100)
        y = stats.norm.pdf(x, stat_result.mean, stat_result.std)
        ax.plot(x, y, 'r-', lw=2, label='Normal fit')
        
        # Add confidence interval
        ax.axvline(stat_result.confidence_interval[0], color='gray',
                  linestyle='--', label='95% CI')
        ax.axvline(stat_result.confidence_interval[1], color='gray',
                  linestyle='--')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.legend()
        
        return ax

def configure_plotting(style: str = 'default',
                      context: str = 'notebook',
                      figsize: Optional[Tuple[int, int]] = None):
    """
    Configure global plotting settings
    
    Args:
        style: matplotlib/seaborn style name
        context: seaborn context name
        figsize: Optional default figure size
    """
    plt.style.use(style)
    sns.set_context(context)
    if figsize is not None:
        plt.rcParams['figure.figsize'] = figsize