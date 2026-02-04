
<img width="617" height="340" alt="logo" src="https://github.com/user-attachments/assets/1ab25208-eaa7-4f07-9992-c7b0e4e51507" />

# Hierarchical-ASI-Supervision

## Overview
This repository provides a rigorous theoretical foundation for deception as constrained optimal control. **Part 1** focuses on modeling and simulating the detectability–capability tradeoff of deceptive agents in reinforcement learning.

## About
The repo contains a Monte Carlo simulation environment to model the `Treacherous Turn` in advanced AI systems. Unlike traditional safety monitors that rely on output thresholds, this framework uses a **Statistical Lie Detector** with **Shannon Entropy** to identify when an agent is masking "Hidden Risk" behind a "Safe" visible policy.

## Features
- KL-regularized policy optimization framework
- Closed-form solution for optimal deceptive policies
- Gradient-based simulations with variable detection pressure (`lambda`)
- Tradeoff visualization: detectability vs hidden objective reward
- Scalable to high-dimensional action spaces

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/noob6t5/Hierarchical-ASI-Supervision.git
cd Hierarchical-ASI-Supervision
python3 supervision.py
```

<img width="800" height="589" alt="detecting" src="https://github.com/user-attachments/assets/7ee7fd01-1329-43b2-b1f5-b83d7d6b5dac" />




<img width="800" height="592" alt="scaling" src="https://github.com/user-attachments/assets/0f82b9ef-ee0f-4e56-8441-ea3905514d42" />




## Citation

Sangharsha Upadhyaya (2026). Detectability Limits of Deceptive Optimization under KL-Regularized Behavioral Constraints (Part 1). Zenodo: https://doi.org/10.5281/zenodo.18485454
