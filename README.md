# PO_mcmc_py: Partial Order Preference Learning with MCMC

A Python implementation for inferring partial orders from preference data using Markov Chain Monte Carlo (MCMC) methods.

## 🚀 Overview

This repository implements the methodology described in [Nicholls et al. (2024)](https://arxiv.org/abs/2212.05524) for partial order inference using MCMC sampling.

### Key Features

- **MCMC Simulation**: Metropolis-Hastings sampling for partial order inference
- **Flexible Modeling**: Support for covariates and multi-dimensional latent spaces
- **Comprehensive Utilities**: Tools for partial order operations, validation, and analysis
- **Interactive Notebooks**: Complete workflow examples with visualizations
- **Robust Implementation**: Well-tested utility functions with proper error handling

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/PREF_PO_PY.git
cd PREF_PO_PY
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Usage

### Quick Start

```python
from src.utils import BasicUtils, StatisticalUtils, GenerationUtils
from src.mcmc import mcmc_partial_order_k
from src.visualization import POPlot

# Generate a partial order
n = 6  # Number of items
eta = StatisticalUtils.generate_latent_positions(n, K=2, rho=0.8)
h_true = BasicUtils.generate_partial_order(eta)

# Visualize
POPlot.visualize_partial_order(h_true, list(range(n)))

# Run MCMC inference
results = mcmc_partial_order_k(observed_orders, config)
```

### Jupyter Notebooks

Explore the complete workflow in our interactive notebooks:
- `notebook/mcmc_simulation with_rj_k.ipynb`: Full MCMC simulation with K-dimensional latent space
- `notebook/mcmc_simulation.ipynb`: Basic MCMC implementation

## 📁 Project Structure

```
PREF_PO_PY/
├── src/
│   ├── utils/           # Core utility functions
│   ├── mcmc/            # MCMC implementation
│   ├── visualization/   # Plotting and visualization
│   └── data/            # Data handling utilities
├── notebook/            # Jupyter notebooks with examples
├── config/              # Configuration files
└── requirements.txt     # Dependencies
```

## 🔧 Core Components

### Utilities (`src/utils/`)
- `BasicUtils`: Partial order operations, validation, linear extensions
- `StatisticalUtils`: Statistical functions and probability calculations  
- `GenerationUtils`: Data generation and sampling utilities

### MCMC (`src/mcmc/`)
- `mcmc_partial_order_k`: Main MCMC simulation with K-dimensional latent space
- `LogLikelihoodCache`: Efficient likelihood computation with caching

### Visualization (`src/visualization/`)
- `POPlot`: Comprehensive plotting tools for partial orders and results

## 📈 Features

- **Transitive Reduction**: Efficient computation of minimal partial orders
- **Linear Extensions**: Generation and counting of valid total orders
- **MCMC Diagnostics**: Convergence analysis and posterior visualization
- **Covariate Support**: Integration of external variables into the model
- **Noise Modeling**: Mallows model for handling preference inconsistencies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{nicholls2024partial,
  title={Partial Order Inference with MCMC},
  author={Nicholls et al.},
  journal={arXiv preprint arXiv:2212.05524},
  year={2024}
}
```

## 🐛 Issues & Support

Please report issues on the [GitHub Issues](https://github.com/YOUR_USERNAME/PREF_PO_PY/issues) page.
