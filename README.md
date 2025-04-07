# Chaotic System Analysis

A Python toolkit for simulating chaotic systems, reconstructing attractors using time-delay embedding, and estimating fractal dimensions.

## Project Overview

This project implements the following workflow:

1. **Simulation**: Simulate a chaotic system (currently supports the Rössler system) and generate time series data.
2. **Reconstruction**: Use time-delay embedding to reconstruct the attractor from a single time series.
3. **Dimension Estimation**: Estimate the correlation dimension of the reconstructed attractor.
4. **Visualization**: Generate detailed plots and dashboards for each simulation run.

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

Install the dependencies:

```bash
pip install numpy scipy matplotlib
```

## Usage

### Basic Usage

Run the main script with default parameters:

```bash
python main.py
```

This will:
1. Simulate the Rössler system with default parameters (a=0.2, b=0.2, c=5.7)
2. Reconstruct the attractor using time-delay embedding
3. Estimate the correlation dimension
4. Generate a dashboard with plots and results in the `results` directory

### Command-Line Arguments

Customize the simulation and analysis with various command-line arguments:

```bash
python main.py --system rossler --a 0.2 --b 0.2 --c 5.7 --t_end 500 --dt 0.01 --embedding_dim 3
```

Key parameters:
- `--system`: The chaotic system to simulate (currently only 'rossler' is supported)
- `--a`, `--b`, `--c`: Parameters for the Rössler system
- `--initial_state`: Initial conditions (comma-separated)
- `--t_start`, `--t_end`: Time span for simulation
- `--dt`: Time step for output
- `--discard_transient`: Time to discard as transient
- `--embedding_var`: Variable to use for embedding (0=x, 1=y, 2=z)
- `--embedding_dim`: Embedding dimension
- `--delay`: Time delay (in indices). If not specified, calculated automatically
- `--results_dir`: Directory to save results

Run `python main.py --help` for a complete list of options.

## Example Results

After running the simulation, results will be saved in a timestamped directory under `results/`. Each run includes:

- Time series plots for each variable
- Phase space plots of the original attractor
- The reconstructed attractor from time-delay embedding
- Mutual information plot (if delay was calculated automatically)
- Correlation dimension estimation plot
- A comprehensive dashboard with all plots and parameters

## Running Tests

Unit tests can be run with:

```bash
python -m unittest test_chaotic_analysis.py
```

## Components

- `chaotic_systems.py`: Implementation of chaotic system ODEs and simulation functions
- `embedding.py`: Time-delay embedding and dimension estimation algorithms
- `visualization.py`: Plotting and results management
- `main.py`: Main script to run the complete workflow
- `test_chaotic_analysis.py`: Unit tests for all components

## Extending the Project

To add support for additional chaotic systems:

1. Add the system's ODE function to `chaotic_systems.py`
2. Update `main.py` to include the new system as an option
3. Add appropriate command-line arguments for the system's parameters

## References

- Takens, F. (1981). "Detecting strange attractors in turbulence"
- Grassberger, P., & Procaccia, I. (1983). "Measuring the strangeness of strange attractors"
- Rössler, O. E. (1976). "An equation for continuous chaos" 