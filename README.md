# 🧬 Multi-locus Co-evolution Simulation

[![GitHub Actions](https://img.shields.io/badge/Request-Simulation-blue)](../../issues/new?template=simulation_request.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A population genetics simulation modeling multi-locus co-evolution with natural selection, mutation events, and frequency-dependent selection pressure.

## 🚀 Run Your Own Simulation

Click the button below to request a simulation with custom parameters:

[![Run Simulation](../../issues/new?template=simulation_request.yml)](../../issues/new?template=simulation_request.yml)

### How it works:
1. Click the button above → Creates a new Issue
2. Fill in your parameters in the form
3. Submit the issue
4. GitHub Actions runs the simulation automatically
5. Results are posted back to your issue as a downloadable zip file

### Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `NUM_GENERATIONS` | Number of generations to simulate | 500 |
| `TOTAL_POP` | Initial population size | 10,000 |
| `C0` | Base fitness (non-selective survival) | 1.0 |
| `λ` | Overall survival rate | 0.8 |
| `LOCI` | Comma-separated locus names | A,B |
| `SAVE_INTERVAL` | Save data every N generations | 100 |

### Locus Parameters (JSON):

```json
{
  "A": {
    "epsilon": 0.1,      // Additive advantage of dominant allele
    "alpha": 0.05,       // Initial selection pressure
    "mu0": 0.0,          // Initial baseline performance
    "mutations": [[10, 100], [5, 200]]  // [[count, generation], ...]
  },
  "B": {
    "epsilon": 0.15,
    "alpha": 0.03,
    "mu0": 0.0,
    "mutations": []
  }
}
