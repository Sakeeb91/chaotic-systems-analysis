# 🌀 Chaotic System Analysis Toolkit

![GitHub](https://img.shields.io/github/license/Sakeeb91/chaotic-systems-analysis?style=flat-square)
![Python](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat-square&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat-square&logo=Matplotlib&logoColor=black)

<p align="center">
  <img src="https://github.com/Sakeeb91/chaotic-systems-analysis/raw/main/docs/images/rossler_3d.png" alt="Rössler Attractor" width="400"/>
</p>

> A comprehensive Python toolkit for simulating chaotic systems, reconstructing attractors using time-delay embedding, and estimating fractal dimensions. This project demonstrates advanced mathematical modeling, data analysis techniques, and scientific visualization.

## 🚀 Features

- **Chaotic System Simulation**: Accurately simulate the Rössler system with configurable parameters
- **Time-Delay Embedding**: Reconstruct higher-dimensional attractors from a single time series
- **Fractal Dimension Estimation**: Calculate the correlation dimension using the Grassberger-Procaccia algorithm
- **Automatic Parameter Selection**: Determine optimal embedding parameters using mutual information
- **Beautiful Visualizations**: Generate high-quality plots with detailed dashboards for each analysis
- **Comprehensive Testing**: Unit tests for all components ensure reliability and correctness

## 📊 Gallery

<p align="center">
  <img src="https://github.com/Sakeeb91/chaotic-systems-analysis/raw/main/docs/images/phase_space.png" alt="Phase Space" width="400"/>
  <img src="https://github.com/Sakeeb91/chaotic-systems-analysis/raw/main/docs/images/embedded_attractor.png" alt="Embedded Attractor" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/Sakeeb91/chaotic-systems-analysis/raw/main/docs/images/mutual_information.png" alt="Mutual Information" width="400"/>
  <img src="https://github.com/Sakeeb91/chaotic-systems-analysis/raw/main/docs/images/dimension_estimation.png" alt="Dimension Estimation" width="400"/>
</p>

## 🧠 Technical Skills Demonstrated

- **Scientific Computing**: Advanced mathematical modeling of nonlinear dynamical systems
- **Signal Processing**: Time-series analysis and reconstruction techniques
- **Data Visualization**: Creating informative, publication-quality visualizations
- **Software Architecture**: Modular, maintainable code organization with separation of concerns
- **Testing**: Comprehensive unit testing with edge case handling
- **Documentation**: Clear documentation with theoretical background and practical usage examples

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/chaotic-systems-analysis.git
cd chaotic-systems-analysis

# Install dependencies
pip install -r requirements.txt
```

## 📝 Usage

### Basic Simulation

```bash
python main.py
```

This will:
1. Simulate the Rössler system with default parameters
2. Reconstruct the attractor using time-delay embedding
3. Estimate the correlation dimension
4. Generate a dashboard with plots and analysis in the `results` directory

### Advanced Usage

```bash
# Change system parameters
python main.py --a 0.1 --b 0.1 --c 14.0

# Modify embedding parameters
python main.py --embedding_dim 4 --embedding_var 1

# Adjust simulation length and transient removal
python main.py --t_end 1000 --discard_transient 200
```

### Example Dashboard

Each run produces a comprehensive dashboard with:

- Time series plots for all variables
- Original 3D attractor visualization
- 2D projections of the phase space
- Reconstructed attractor from embedding
- Mutual information analysis for optimal delay selection
- Correlation dimension estimation with scaling region

## 📂 Project Structure

```
chaotic-systems-analysis/
├── chaotic_systems.py    # ODE implementations and simulation functions
├── embedding.py          # Time-delay embedding and dimension estimation
├── visualization.py      # Plotting and results management
├── main.py               # Command-line interface and workflow
├── test_chaotic_analysis.py  # Comprehensive unit tests
└── requirements.txt      # Dependencies
```

## 🔍 Core Components

### Chaotic Systems Module

Implements differential equations for chaotic systems and provides simulation capabilities:

```python
# Rössler system with customizable parameters
def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx_dt = -y - z
    dy_dt = x + a * y
    dz_dt = b + z * (x - c)
    return [dx_dt, dy_dt, dz_dt]
```

### Embedding Module

Provides tools for attractor reconstruction and dimension estimation:

```python
# Time-delay embedding to reconstruct attractor
embedded_data = time_delay_embedding(time_series, embedding_dim, delay)

# Optimal delay selection via mutual information
optimal_delay = find_optimal_delay(time_series)

# Correlation dimension estimation
dim_results = estimate_correlation_dimension(embedded_data)
```

### Visualization Module

Generates beautiful plots and comprehensive dashboards:

```python
# Initialize results manager for a run
results_mgr = ResultsManager()

# Plot time series data
results_mgr.plot_time_series(time, values)

# Plot 3D phase space
results_mgr.plot_phase_space(states)

# Create correlation dimension plot
results_mgr.plot_correlation_dimension(dim_results, embedding_dim)
```

## 🔮 Future Enhancements

- Add support for additional chaotic systems (Lorenz, Duffing, etc.)
- Implement Lyapunov exponent estimation
- Create automated parameter sweeps to study bifurcations
- Add interactive 3D visualizations with Plotly
- Develop a web interface for result exploration

## 📚 Academic Background

This project implements concepts from:

- **Nonlinear Dynamics**: Chaotic systems and strange attractors
- **Embedding Theory**: Takens' embedding theorem
- **Fractal Geometry**: Dimension estimation of strange attractors
- **Time Series Analysis**: Reconstruction of dynamics from observational data

## 🔗 References

- Takens, F. (1981). "Detecting strange attractors in turbulence"
- Grassberger, P., & Procaccia, I. (1983). "Measuring the strangeness of strange attractors"
- Rössler, O. E. (1976). "An equation for continuous chaos"

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  <i>Created for analyzing chaotic electronic circuit data</i><br>
  <i>© 2023 Shafkat Rahman</i>
</p> 