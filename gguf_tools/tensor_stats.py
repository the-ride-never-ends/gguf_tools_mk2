"""Structural analysis of neural network weight tensors.

This module provides tools for analyzing weight matrices from neural networks,
focusing on spectral properties, rank structure, correlations, and potential
fractal-like characteristics.

# Weight Tensor Structural Analysis Toolkit

Comprehensive tools for analyzing structural properties of neural network weight tensors, with focus on spectral characteristics, rank structure, spatial correlations, and potential fractal-like behavior.

## Overview

This toolkit provides rigorous mathematical analysis for weight matrices from neural networks, particularly useful for:
- Understanding learned representations in attention layers
- Detecting hierarchical structure in weights
- Identifying low-rank patterns
- Analyzing spatial correlations
- Checking for scale-invariant (fractal-like) properties

## Installation

Required packages:
```bash
pip install numpy scipy matplotlib
pip install PyWavelets  # Optional, for wavelet analysis
```

## Quick Start

### Analyze Synthetic Weights
```python
from weight_tensor_analysis import analyze_weight_tensor
from weight_tensor_visualization import create_full_report
import numpy as np

# Generate or load weights
weights = np.random.normal(0, 0.01, (640, 2048)).astype(np.float32)

# Run analysis
results = analyze_weight_tensor(weights, run_wavelet=True)

# Generate report and visualizations
summary, figures = create_full_report(results, tensor_name='output')
print(summary)
```

### Analyze Real Weights
```python
# From command line
python example_usage.py path/to/weights.npy

# Or in code
from example_usage import analyze_real_weights
analyze_real_weights('path/to/weights.npy')
```

## Analysis Methods

### 1. Power Spectral Density Analysis

Computes 2D Fourier transform and analyzes frequency content.

**What it detects:**
- Power-law scaling: S(f) ∝ f^(-β)
- Self-affine fractals show linear log-log power spectrum
- Slope β relates to Hurst exponent and fractal dimension

**Interpretation:**
- β < -1: Strong long-range correlations, potential fractal structure
- β ≈ 0: White noise, no correlations
- High R^2 (>0.95): Consistent power-law behavior

**Code:**
```python
from weight_tensor_analysis import compute_power_spectrum

result = compute_power_spectrum(weights)
print(f"Spectral slope: {result['slope']:.3f}")
print(f"Power-law detected: {result['is_powerlaw']}")
```

### 2. Singular Value Decomposition Analysis

Decomposes weight matrix and analyzes singular value decay.

**What it detects:**
- Low-rank structure: σᵢ ∝ i^(-α)
- Effective rank via entropy measure
- Variance distribution across components

**Interpretation:**
- Low effective rank (<30% of dimension): Structured, compressible weights
- Power-law decay: Hierarchical importance of singular vectors
- High R^2: Scale-invariant structure

**Code:**
```python
from weight_tensor_analysis import compute_svd_spectrum

result = compute_svd_spectrum(weights)
print(f"Effective rank: {result['effective_rank']:.1f}")
print(f"90% variance captured by {result['rank_90']} components")
```

### 3. Spatial Correlation Analysis

Measures autocorrelation at different spatial lags.

**What it detects:**
- Correlation decay length (1/e point)
- Anisotropy (different horizontal vs vertical correlations)
- Range of spatial dependencies

**Interpretation:**
- Short decay (<10% of dimension): Local structure
- Long decay (>50% of dimension): Global dependencies
- Asymmetric decay: Directional biases in learned features

**Code:**
```python
from weight_tensor_analysis import compute_spatial_correlations

result = compute_spatial_correlations(weights)
print(f"Horizontal decay: {result['h_decay_length']:.1f}")
print(f"Vertical decay: {result['v_decay_length']:.1f}")
```

### 4. Wavelet Energy Analysis

Multi-scale decomposition examining energy distribution across scales.

**What it detects:**
- Hierarchical energy distribution
- Scale-dependent feature strength
- Power-law scaling of energy

**Interpretation:**
- Uniform energy: No preferred scales
- Power-law energy scaling: Self-similar structure
- Concentrated energy at specific scales: Characteristic length scales

**Code:**
```python
from weight_tensor_analysis import compute_wavelet_energy

result = compute_wavelet_energy(weights)
print(f"Energy scaling slope: {result['slope']:.3f}")
```

## Understanding the Results

### Fractal Indicators

Strong evidence for fractal-like structure requires:
1. **Power spectrum**: Linear in log-log space with R^2 > 0.95 and slope < -0.5
2. **SVD spectrum**: Power-law decay with R^2 > 0.95 and slope < -0.2
3. **Consistency**: Multiple methods showing scale-invariant behavior

### Typical Neural Network Weights

Most trained weights exhibit:
- Approximately Gaussian value distribution
- Moderate rank (30-70% effective rank ratio)
- Short-range correlations (decay length < 20% of dimension)
- No clear power-law scaling

This is normal and indicates well-regularized learning.

### Unusual Patterns

Fractal-like or strongly scale-invariant weights suggest:
- Hierarchical feature organization
- Critical dynamics during training
- Potential overparameterization
- Novel architectural properties

## Caveats and Limitations

### Scale Range Limitations
- Raster resolution bounds effective scale range
- Need 2-3 orders of magnitude for reliable dimension estimates
- Typical matrices: 1-2 orders of magnitude maximum

### Noise Contamination
- Random noise can mimic fractal signatures
- Measurement noise vs signal becomes indistinguishable at small scales
- Preprocessing affects results significantly

### Finite-Size Effects
- Boundary conditions introduce systematic bias
- Statistical fluctuations larger for smaller tensors
- Grid alignment affects box-counting results

### False Positives
- Many non-fractal processes show power-law-like behavior over limited ranges
- Texture and hierarchical structure can mimic fractals
- Anisotropic patterns complicate analysis

### Parameter Sensitivity
- Threshold choices affect results
- Box size sequences matter for box-counting
- Regression range selection impacts slope estimates
- Small changes can shift dimension estimates by 0.2-0.5

## Practical Recommendations

1. **Use multiple methods**: Don't rely on single metric
2. **Check consistency**: Different methods should agree
3. **Validate on synthetics**: Test with known fractal patterns
4. **Report uncertainties**: Include R^2 and confidence intervals
5. **Consider alternatives**: Is it really fractals or just hierarchical structure?
6. **Compare layers**: Analyze multiple layers to identify patterns
7. **Track over training**: Monitor how structure evolves

## Example: Diffusion Model Attention Weights

Given statistics from `model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn2.to_v.weight`:
- Shape: (640, 2048)
- Mean: -8.2e-6 (near-zero)
- Std: 0.0107 (small, controlled)
- Range: [-0.159, 0.124]

These statistics suggest:
- Well-initialized or well-trained weights
- Centered distribution (mean ≈ 0)
- Controlled magnitude (small std)
- Symmetric range

Analysis would likely show:
- Approximately Gaussian distribution (no strong fractality)
- Moderate effective rank
- Short to medium-range correlations
- No clear power-law scaling

This is typical for attention weights in diffusion models.

## Visualization Outputs

The toolkit generates:

1. **Spectral Analysis Plot**
   - 2D power spectrum heatmap
   - Radial power spectrum (log-log)
   - Linear fit for power-law detection

2. **SVD Analysis Plot**
   - Singular value decay
   - Cumulative variance explained
   - Log-log decay with fit

3. **Correlation Analysis Plot**
   - Horizontal autocorrelation
   - Vertical autocorrelation
   - Decay length markers

4. **Wavelet Analysis Plot**
   - Energy distribution across scales
   - Log-log energy scaling

5. **Text Report**
   - Summary statistics
   - Fractal indicators
   - Interpretation guidelines

## File Structure

```
weight_tensor_analysis.py         # Core analysis functions
weight_tensor_visualization.py    # Plotting and reporting
example_usage.py                  # Usage examples and demos
README.md                         # This file
```

## Advanced Usage

### Custom Analysis Pipeline

```python
from weight_tensor_analysis import (
    compute_power_spectrum,
    compute_svd_spectrum,
    compute_spatial_correlations
)

weights = load_your_weights()

spectral = compute_power_spectrum(
    weights,
    min_freq_percentile=10.0,
    max_freq_percentile=90.0
)

svd = compute_svd_spectrum(
    weights,
    variance_threshold=0.95
)

corr = compute_spatial_correlations(
    weights,
    max_lag=100
)
```

### Comparing Multiple Layers

```python
from weight_tensor_analysis import analyze_weight_tensor

layers = {
    'attention_q': load_weights('attn_q.npy'),
    'attention_k': load_weights('attn_k.npy'),
    'attention_v': load_weights('attn_v.npy'),
}

results = {}
for name, weights in layers.items():
    results[name] = analyze_weight_tensor(weights)
    print(f"\n{name}:")
    print(f"  Effective rank: {results[name]['svd']['effective_rank']:.1f}")
    print(f"  Spectral slope: {results[name]['spectral']['slope']:.3f}")
```

## References

- Fractal dimension estimation: Falconer, K. (2003). Fractal Geometry
- Power spectral methods: Mandelbrot & Van Ness (1968). Fractional Brownian motions
- Neural network analysis: Martin & Mahoney (2019). Traditional and Heavy-Tailed Self Regularization
"""
import numpy as np
import numpy.typing as npt
from scipy import fft
from scipy.stats import linregress
from typing import TypedDict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path


from gguf_visualizers.tensor_to_image import TensorToImage
from config.config import PROJECT_ROOT


project_path = Path(PROJECT_ROOT)

try:
    import pywt
except ImportError:
    raise ImportError("pywt required for wavelet analysis. Install via: pip install PyWavelets")


THIS_FILE = Path(__file__)
THIS_DIR = THIS_FILE.parent
MAIN_DIR = THIS_DIR.parent


class SpectralResult(TypedDict):
    """Results from power spectral density analysis."""
    freq_y: npt.NDArray[np.float32]  # <-- Split into two lines
    freq_x: npt.NDArray[np.float32]  # <--
    power_spectrum: npt.NDArray[np.float32]
    radial_freq: npt.NDArray[np.float32]
    radial_power: npt.NDArray[np.float32]
    slope: float
    r_squared: float
    is_powerlaw: bool


class SVDResult(TypedDict):
    """Results from singular value decomposition analysis."""
    singular_values: npt.NDArray[np.float32]
    normalized_sv: npt.NDArray[np.float32]
    effective_rank: float
    rank_90: int
    slope: float
    r_squared: float
    is_powerlaw: bool


class CorrelationResult(TypedDict):
    """Results from spatial correlation analysis."""
    horizontal_corr: npt.NDArray[np.float32]
    vertical_corr: npt.NDArray[np.float32]
    lags: npt.NDArray[np.int32]
    h_decay_length: float
    v_decay_length: float


class WaveletResult(TypedDict):
    """Results from wavelet decomposition analysis."""
    scales: list[int]
    energies: npt.NDArray[np.float32]
    normalized_energies: npt.NDArray[np.float32]
    slope: float
    r_squared: float


def compute_power_spectrum(
    weights: npt.NDArray[np.float32],
    min_freq_percentile: float = 5.0,
    max_freq_percentile: float = 95.0
) -> SpectralResult:
    """Compute 2D power spectral density and check for power-law scaling.
    
    Args:
        weights: 2D weight matrix to analyze.
        min_freq_percentile: Minimum frequency percentile for power-law fit.
        max_freq_percentile: Maximum frequency percentile for power-law fit.
    
    Returns:
        Dictionary containing frequencies, power spectrum, radial averages,
        and power-law fit statistics.
    """
    weights_centered = weights - np.mean(weights)
    
    fft2d = fft.fft2(weights_centered)
    power = np.abs(fft2d) ** 2
    power_shifted = fft.fftshift(power)
    
    freq_y = fft.fftshift(fft.fftfreq(weights.shape[0]))
    freq_x = fft.fftshift(fft.fftfreq(weights.shape[1]))
    
    center_y, center_x = np.array(power_shifted.shape) // 2
    y_grid, x_grid = np.ogrid[:power_shifted.shape[0], :power_shifted.shape[1]]
    
    distances = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    max_radius = int(min(center_y, center_x) * 0.9)
    
    radial_power = np.zeros(max_radius)
    radial_counts = np.zeros(max_radius)
    
    for radius in range(1, max_radius):
        mask = (distances >= radius) & (distances < radius + 1)
        if np.any(mask):
            radial_power[radius] = np.mean(power_shifted[mask])
            radial_counts[radius] = np.sum(mask)
    
    radial_freq = np.arange(1, max_radius, dtype=np.float32)
    radial_power = radial_power[1:max_radius]
    
    valid_idx = (radial_power > 0) & np.isfinite(radial_power)
    log_freq = np.log10(radial_freq[valid_idx])
    log_power = np.log10(radial_power[valid_idx])
    
    freq_min = np.percentile(log_freq, min_freq_percentile)
    freq_max = np.percentile(log_freq, max_freq_percentile)
    fit_mask = (log_freq >= freq_min) & (log_freq <= freq_max)
    
    slope, intercept, r_value, _, _ = linregress(
        log_freq[fit_mask], 
        log_power[fit_mask]
    )
    r_squared = r_value ** 2
    
    is_powerlaw = r_squared > 0.95 and slope < -0.5
    
    return SpectralResult(
        freq_y=freq_y.astype(np.float32),
        freq_x=freq_x.astype(np.float32),
        power_spectrum=power_shifted.astype(np.float32),
        radial_freq=radial_freq,
        radial_power=radial_power,
        slope=float(slope),
        r_squared=float(r_squared),
        is_powerlaw=is_powerlaw
    )



def compute_svd_spectrum(
    weights: npt.NDArray[np.float32],
    variance_threshold: float = 0.9,
    min_sv_percentile: float = 5.0,
    max_sv_percentile: float = 95.0
) -> SVDResult:
    """Compute singular value decomposition and analyze spectrum.
    
    Args:
        weights: 2D weight matrix to analyze.
        variance_threshold: Threshold for computing effective rank.
        min_sv_percentile: Minimum singular value percentile for power-law fit.
        max_sv_percentile: Maximum singular value percentile for power-law fit.
    
    Returns:
        Dictionary containing singular values, effective rank metrics,
        and power-law fit statistics.
    """
    _, singular_values, _ = np.linalg.svd(weights, full_matrices=False)
    
    sv_squared = singular_values ** 2
    total_variance = np.sum(sv_squared)
    normalized_sv = singular_values / singular_values[0]
    
    cumsum_variance = np.cumsum(sv_squared)
    rank_90 = int(np.argmax(cumsum_variance >= variance_threshold * total_variance) + 1)
    
    entropy = -np.sum(
        (sv_squared / total_variance) * np.log(sv_squared / total_variance + 1e-10)
    )
    effective_rank = np.exp(entropy)
    
    indices = np.arange(1, len(singular_values) + 1, dtype=np.float32)
    log_indices = np.log10(indices)
    log_sv = np.log10(singular_values)
    
    idx_min = np.percentile(log_indices, min_sv_percentile)
    idx_max = np.percentile(log_indices, max_sv_percentile)
    fit_mask = (log_indices >= idx_min) & (log_indices <= idx_max)
    
    slope, intercept, r_value, _, _ = linregress(
        log_indices[fit_mask],
        log_sv[fit_mask]
    )
    r_squared = r_value ** 2
    
    is_powerlaw = r_squared > 0.95 and slope < -0.2
    
    return SVDResult(
        singular_values=singular_values,
        normalized_sv=normalized_sv,
        effective_rank=float(effective_rank),
        rank_90=rank_90,
        slope=float(slope),
        r_squared=float(r_squared),
        is_powerlaw=is_powerlaw
    )


def compute_spatial_correlations(
    weights: npt.NDArray[np.float32],
    max_lag: int | None = None
) -> CorrelationResult:
    """Compute spatial autocorrelation functions.
    
    Args:
        weights: 2D weight matrix to analyze.
        max_lag: Maximum lag for correlation computation. If None, uses min dimension / 4.
    
    Returns:
        Dictionary containing horizontal and vertical autocorrelations,
        lags, and characteristic decay lengths.
    """
    if max_lag is None:
        max_lag = min(weights.shape) // 4
    
    weights_centered = weights - np.mean(weights)
    variance = np.var(weights)
    
    h_corr = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            h_corr[lag] = 1.0
        else:
            overlapping = weights_centered[:, :-lag] * weights_centered[:, lag:]
            h_corr[lag] = np.mean(overlapping) / variance
    
    v_corr = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            v_corr[lag] = 1.0
        else:
            overlapping = weights_centered[:-lag, :] * weights_centered[lag:, :]
            v_corr[lag] = np.mean(overlapping) / variance
    
    h_decay = _compute_decay_length(h_corr)
    v_decay = _compute_decay_length(v_corr)
    
    lags = np.arange(max_lag, dtype=np.int32)
    
    return CorrelationResult(
        horizontal_corr=h_corr.astype(np.float32),
        vertical_corr=v_corr.astype(np.float32),
        lags=lags,
        h_decay_length=float(h_decay),
        v_decay_length=float(v_decay)
    )


def _compute_decay_length(corr: npt.NDArray[np.float32]) -> float:
    """Compute correlation decay length as 1/e point."""
    threshold = 1.0 / np.e
    try:
        idx = np.argmax(corr < threshold)
        if idx == 0 and corr[0] >= threshold:
            return float(len(corr))
        return float(idx)
    except (ValueError, IndexError):
        return float(len(corr))


def compute_wavelet_energy(
    weights: npt.NDArray[np.float32],
    wavelet: str = 'haar'
) -> WaveletResult:
    """Compute multi-scale wavelet decomposition energy distribution.
    
    Args:
        weights: 2D weight matrix to analyze.
        wavelet: Wavelet type for decomposition.
    
    Returns:
        Dictionary containing scales, energies, and power-law fit statistics.
    """
    max_level = pywt.dwt_max_level(min(weights.shape), wavelet)
    max_level = min(max_level, 6)
    
    coeffs = pywt.wavedec2(weights, wavelet, level=max_level)
    
    energies = []
    scales = []
    
    approximation = coeffs[0]
    energies.append(float(np.sum(approximation ** 2)))
    scales.append(2 ** max_level)
    
    for level, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        scale = 2 ** (max_level - level + 1)
        energy = float(np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2))
        energies.append(energy)
        scales.append(scale)

    energies_array = np.array(energies, dtype=np.float32)
    total_energy = np.sum(energies_array)
    normalized = energies_array / total_energy

    log_scales = np.log10(scales)
    log_energies = np.log10(energies_array)

    slope, intercept, r_value, _, _ = linregress(log_scales, log_energies)
    r_squared = r_value ** 2

    return WaveletResult(
        scales=scales,
        energies=energies_array,
        normalized_energies=normalized,
        slope=float(slope),
        r_squared=float(r_squared)
    )


def analyze_weight_tensor(
    weights: npt.NDArray[np.float32],
    run_wavelet: bool = False
) -> dict:
    """Perform comprehensive structural analysis on weight tensor.
    
    Args:
        weights: 2D weight matrix to analyze.
        run_wavelet: Whether to include wavelet analysis (requires pywt).
    
    Returns:
        Dictionary containing all analysis results.
    """
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {weights.shape}")
    
    results = {
        'shape': weights.shape,
        'basic_stats': {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights))
        }
    }
    
    results['spectral'] = compute_power_spectrum(weights)
    print("Calculated power spectrum")
    results['svd'] = compute_svd_spectrum(weights)
    print("Calculated SVD spectrum")
    results['spatial_correlation'] = compute_spatial_correlations(weights)
    print("Calculated spatial correlations")
    results['moran_i'] = calculate_moran_index(weights)
    print("Calculated Moran's I")

    if run_wavelet:
        try:
            results['wavelet'] = compute_wavelet_energy(weights)
            print("Calculated wavelet energy")
        except ImportError as e:
            results['wavelet'] = {'error': str(e)}
    
    return results





def plot_spectral_analysis(
    result: dict[str, Any],
    figsize: tuple[int, int] = (15, 5)
) -> Figure:
    """Create plots for power spectral density analysis.
    
    Args:
        result: SpectralResult dictionary from compute_power_spectrum.
        figsize: Figure size as (width, height).
    
    Returns:
        Matplotlib figure with three subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    power_db = 10 * np.log10(result['power_spectrum'] + 1e-10)
    im = axes[0].imshow(power_db, cmap='viridis', aspect='auto')
    axes[0].set_title('2D Power Spectrum (dB)')
    axes[0].set_xlabel('Frequency X')
    axes[0].set_ylabel('Frequency Y')
    plt.colorbar(im, ax=axes[0])
    
    axes[1].loglog(result['radial_freq'][1:], result['radial_power'][1:], 'b-', alpha=0.6)
    axes[1].set_xlabel('Radial Frequency')
    axes[1].set_ylabel('Power')
    axes[1].set_title(f"Radial Power Spectrum\nSlope: {result['slope']:.3f}, R^2: {result['r_squared']:.3f}")
    axes[1].grid(True, alpha=0.3)
    
    if result['is_powerlaw']:
        axes[1].text(0.05, 0.95, 'Power-law detected', 
                    transform=axes[1].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    log_freq = np.log10(result['radial_freq'][1:])
    log_power = np.log10(result['radial_power'][1:])
    axes[2].plot(log_freq, log_power, 'b.', alpha=0.5, markersize=3)
    
    fit_line_x = np.array([log_freq.min(), log_freq.max()])
    fit_line_y = result['slope'] * fit_line_x + (log_power[0] - result['slope'] * log_freq[0])
    axes[2].plot(fit_line_x, fit_line_y, 'r--', linewidth=2, label='Linear fit')
    
    axes[2].set_xlabel('log₁₀(Frequency)')
    axes[2].set_ylabel('log₁₀(Power)')
    axes[2].set_title('Log-Log Power Spectrum')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_svd_analysis(
    result: dict[str, Any],
    figsize: tuple[int, int] = (15, 5)
) -> Figure:
    """Create plots for singular value decomposition analysis.
    
    Args:
        result: SVDResult dictionary from compute_svd_spectrum.
        figsize: Figure size as (width, height).
    
    Returns:
        Matplotlib figure with three subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    indices = np.arange(1, len(result['singular_values']) + 1)
    axes[0].semilogy(indices, result['singular_values'], 'b-', linewidth=1.5)
    axes[0].axhline(result['singular_values'][result['rank_90']], 
                   color='r', linestyle='--', alpha=0.5, label=f"90% variance (rank {result['rank_90']})")
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Singular Value')
    axes[0].set_title(f"Singular Value Spectrum\nEffective Rank: {result['effective_rank']:.1f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    cumsum = np.cumsum(result['singular_values']**2) / np.sum(result['singular_values']**2)
    axes[1].plot(indices, cumsum, 'g-', linewidth=2)
    axes[1].axhline(0.9, color='r', linestyle='--', alpha=0.5)
    axes[1].axvline(result['rank_90'], color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance')
    axes[1].grid(True, alpha=0.3)
    
    log_indices = np.log10(indices)
    log_sv = np.log10(result['singular_values'])
    axes[2].plot(log_indices, log_sv, 'b.', alpha=0.5, markersize=3)
    
    fit_line_x = np.array([log_indices.min(), log_indices.max()])
    fit_line_y = result['slope'] * fit_line_x + (log_sv[0] - result['slope'] * log_indices[0])
    axes[2].plot(fit_line_x, fit_line_y, 'r--', linewidth=2, label='Linear fit')
    
    axes[2].set_xlabel('log₁₀(Index)')
    axes[2].set_ylabel('log₁₀(Singular Value)')
    axes[2].set_title(f"Log-Log SV Decay\nSlope: {result['slope']:.3f}, R^2: {result['r_squared']:.3f}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if result['is_powerlaw']:
        axes[2].text(0.05, 0.95, 'Power-law detected', 
                    transform=axes[2].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    return fig


def plot_correlation_analysis(
    result: dict[str, Any],
    figsize: tuple[int, int] = (12, 5)
) -> Figure:
    """Create plots for spatial correlation analysis.
    
    Args:
        result: CorrelationResult dictionary from compute_spatial_correlations.
        figsize: Figure size as (width, height).
    
    Returns:
        Matplotlib figure with two subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].plot(result['lags'], result['horizontal_corr'], 'b-', linewidth=2, label='Horizontal')
    axes[0].axhline(1/np.e, color='r', linestyle='--', alpha=0.5, label='1/e threshold')
    axes[0].axvline(result['h_decay_length'], color='b', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].set_title(f"Horizontal Correlation\nDecay length: {result['h_decay_length']:.1f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.1, 1.1])
    
    axes[1].plot(result['lags'], result['vertical_corr'], 'g-', linewidth=2, label='Vertical')
    axes[1].axhline(1/np.e, color='r', linestyle='--', alpha=0.5, label='1/e threshold')
    axes[1].axvline(result['v_decay_length'], color='g', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title(f"Vertical Correlation\nDecay length: {result['v_decay_length']:.1f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    return fig


def plot_wavelet_analysis(
    result: dict[str, Any],
    figsize: tuple[int, int] = (12, 5)
) -> Figure:
    """Create plots for wavelet energy analysis.
    
    Args:
        result: WaveletResult dictionary from compute_wavelet_energy.
        figsize: Figure size as (width, height).
    
    Returns:
        Matplotlib figure with two subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].bar(range(len(result['scales'])), result['normalized_energies'], 
               color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Scale Level')
    axes[0].set_ylabel('Normalized Energy')
    axes[0].set_title('Energy Distribution Across Scales')
    axes[0].set_xticks(range(len(result['scales'])))
    axes[0].set_xticklabels([f"{s}" for s in result['scales']])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    log_scales = np.log10(result['scales'])
    log_energies = np.log10(result['energies'])
    axes[1].plot(log_scales, log_energies, 'bo-', linewidth=2, markersize=8, alpha=0.7)
    
    fit_line_x = np.array([log_scales.min(), log_scales.max()])
    fit_line_y = result['slope'] * fit_line_x + (log_energies[0] - result['slope'] * log_scales[0])
    axes[1].plot(fit_line_x, fit_line_y, 'r--', linewidth=2, label='Linear fit')
    
    axes[1].set_xlabel('log₁₀(Scale)')
    axes[1].set_ylabel('log₁₀(Energy)')
    axes[1].set_title(f"Energy Scaling\nSlope: {result['slope']:.3f}, R^2: {result['r_squared']:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_moran_index(
    data: npt.NDArray[np.float32],
    weight_type: str = 'rook'
) -> float:
    """Calculate Moran's I spatial autocorrelation index.
    
    Args:
        data: 2D array of values to analyze.
        weight_type: Spatial weight scheme ('rook' or 'queen').
    
    Returns:
        Moran's I value. Range typically [-1, 1] where:
        - I > 0: positive spatial autocorrelation (clustering)
        - I ≈ 0: random spatial pattern
        - I < 0: negative spatial autocorrelation (dispersion)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    
    rows, cols = data.shape
    n = rows * cols
    
    mean_val = np.mean(data)
    deviations = data - mean_val
    
    numerator = 0.0
    denominator = np.sum(deviations ** 2)
    total_weight = 0.0
    
    for i in range(rows):
        for j in range(cols):
            neighbors = _get_neighbors(i, j, rows, cols, weight_type)
            
            for ni, nj in neighbors:
                weight = 1.0
                numerator += weight * deviations[i, j] * deviations[ni, nj]
                total_weight += weight
    
    if total_weight == 0 or denominator == 0:
        return 0.0
    
    moran_i = (n / total_weight) * (numerator / denominator)
    
    return float(moran_i)


def _get_neighbors(
    i: int,
    j: int,
    rows: int,
    cols: int,
    weight_type: str
) -> list[tuple[int, int]]:
    """Get neighbor indices for a cell."""
    neighbors = []
    
    if weight_type == 'rook':
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif weight_type == 'queen':
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    for di, dj in deltas:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            neighbors.append((ni, nj))
    
    return neighbors




def generate_summary_report(results: dict[str, Any]) -> str:
    """Generate text summary of analysis results.
    
    Args:
        results: Complete results dictionary from analyze_weight_tensor.
    
    Returns:
        Formatted summary report string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("WEIGHT TENSOR STRUCTURAL ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("BASIC STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Shape: {results['shape']}")
    stats = results['basic_stats']
    lines.append(f"Mean: {stats['mean']:.6e}")
    lines.append(f"Std Dev: {stats['std']:.6e}")
    lines.append(f"Range: [{stats['min']:.6e}, {stats['max']:.6e}]")
    lines.append("")
    
    spec = results['spectral']
    lines.append("SPECTRAL ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Power spectrum slope (beta): {spec['slope']:.4f}")
    lines.append(f"R^2 of power-law fit: {spec['r_squared']:.4f}")
    lines.append(f"Power-law behavior: {'YES' if spec['is_powerlaw'] else 'NO'}")
    if spec['is_powerlaw']:
        lines.append(f"  * Suggests fractal-like structure with scaling exponent beta = {spec['slope']:.3f}")
    else:
        lines.append("  * No clear power-law scaling detected")
    lines.append("")
    
    svd = results['svd']
    lines.append("SINGULAR VALUE ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Effective rank: {svd['effective_rank']:.2f} / {results['shape'][0]}")
    lines.append(f"Rank for 90% variance: {svd['rank_90']}")
    lines.append(f"SV decay slope (alpha): {svd['slope']:.4f}")
    lines.append(f"R^2 of power-law fit: {svd['r_squared']:.4f}")
    lines.append(f"Power-law decay: {'YES' if svd['is_powerlaw'] else 'NO'}")
    
    rank_ratio = svd['effective_rank'] / min(results['shape'])
    if rank_ratio < 0.3:
        lines.append("  * Low-rank structure detected (effective rank < 30% of dimension)")
    elif rank_ratio > 0.7:
        lines.append("  * Full-rank behavior (effective rank > 70% of dimension)")
    else:
        lines.append("  * Moderate rank structure")
    lines.append("")
    
    corr = results['spatial_correlation']
    lines.append("SPATIAL CORRELATION ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Horizontal decay length: {corr['h_decay_length']:.2f} units")
    lines.append(f"Vertical decay length: {corr['v_decay_length']:.2f} units")
    
    max_lag = len(corr['lags'])
    if max(corr['h_decay_length'], corr['v_decay_length']) >= max_lag * 0.9:
        lines.append("  * Long-range correlations present")
    elif max(corr['h_decay_length'], corr['v_decay_length']) < max_lag * 0.1:
        lines.append("  * Short-range correlations (rapid decorrelation)")
    else:
        lines.append("  * Medium-range correlations")
    lines.append("")
    
    if 'wavelet' in results and 'error' not in results['wavelet']:
        wav = results['wavelet']
        lines.append("WAVELET ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Energy scaling slope: {wav['slope']:.4f}")
        lines.append(f"R^2 of scaling fit: {wav['r_squared']:.4f}")
        lines.append(f"Number of scales analyzed: {len(wav['scales'])}")
        lines.append("")
    
    moran = results['moran_i']
    lines.append("MORAN'S I SPATIAL AUTOCORRELATION")
    lines.append(f"Moran's I: {moran:.4f}")
    lines.append("  * Positive values indicate clustering, negative values indicate dispersion")
    lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 80)
    
    fractal_indicators = 0
    if spec['is_powerlaw']:
        fractal_indicators += 1
    if svd['is_powerlaw']:
        fractal_indicators += 1
    
    if fractal_indicators >= 2:
        lines.append("Strong evidence of fractal-like or scale-invariant structure:")
        lines.append("  * Power-law scaling in frequency domain")
        lines.append("  * Power-law decay in singular value spectrum")
        lines.append("")
        lines.append("This suggests:")
        lines.append("  * Hierarchical feature organization")
        lines.append("  * Self-similar patterns across scales")
        lines.append("  * Possible critical learning dynamics")
    elif fractal_indicators == 1:
        lines.append("Moderate evidence of scale-invariant structure:")
        if spec['is_powerlaw']:
            lines.append("  * Power-law scaling in frequency domain")
        if svd['is_powerlaw']:
            lines.append("  * Power-law decay in singular values")
        lines.append("")
        lines.append("This suggests some hierarchical organization but not full fractality.")
    else:
        lines.append("No strong evidence of fractal structure:")
        lines.append("  * Weight distribution appears approximately Gaussian")
        lines.append("  * No clear power-law scaling in analyzed metrics")
        lines.append("")
        lines.append("This is typical for well-trained neural network weights.")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def create_full_report(
    results: dict[str, Any],
    tensor_name: str | None = None
) -> tuple[str, list[Figure]]:
    """Generate complete analysis report with visualizations.
    
    Args:
        results: Complete results dictionary from analyze_weight_tensor.
        tensor_name: Optional path prefix for saving figures.
    
    Returns:
        Tuple of (summary_text, list_of_figures).
    """
    summary = generate_summary_report(results)
    
    figures = [
        plot_spectral_analysis(results['spectral']),
        plot_svd_analysis(results['svd']),
        plot_correlation_analysis(results['spatial_correlation']),
    ]

    if 'wavelet' in results and 'error' not in results['wavelet']:
        figures.append(plot_wavelet_analysis(results['wavelet']))

    save_path = project_path / "output" / tensor_name

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    kwargs = {"dpi": 150, "bbox_inches": "tight"}
    if tensor_name:
        figures[0].savefig(save_path / f"{tensor_name}_spectral.png", **kwargs)
        figures[1].savefig(save_path / f"{tensor_name}_svd.png", **kwargs)
        figures[2].savefig(save_path / f"{tensor_name}_correlation.png", **kwargs)
        if len(figures) > 3:
            figures[3].savefig(save_path / f"{tensor_name}_wavelet.png", **kwargs)

        with open(save_path / f"{tensor_name}_report.txt", 'w') as f:
            f.write(summary)

    return summary, figures









class TensorStats(TensorToImage):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def tensor_stats(self):
        tensor = self._extract_tensor_from_model()

        # Run analysis
        results = analyze_weight_tensor(tensor, run_wavelet=True)
        print("Got results. Generating report...")

        # Generate report and visualizations
        summary, _ = create_full_report(results, tensor_name=self.tensor_name)
        print(summary)
