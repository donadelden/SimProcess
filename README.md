# SimProcess: High Fidelity Simulation of Noisy ICS Physical Processes

This project provides tools for generating synthetic power grid data, processing real power grid measurements, and discriminating between real and simulated scenarios using machine learning.

This code supports a paper accepted at the _11th ACM Cyber-Physical System Security Workshop (CPSS'25)_. The preprint is available on [Arxiv](http://arxiv.org/abs/2505.22638). 

Cite this repository as follows:
```bibtex
@inproceedings{donadel2025simprocess,
  title={SimProcess: High Fidelity Simulation of Noisy ICS Physical Processes},
  author={Donadel, Denis and Crestanello, Gabriele and Morandini, Giulio and Antonioli, Daniele and Conti, Mauro and Merro, Massimo},
  booktitle={Proceedings of the 11th ACM Cyber-Physical System Security Workshop},
  pages={1--12},
  year={2025},
  organization={ACM Press}
}
```

## Project Overview

SimProcess is a comprehensive framework designed to work with power grid data from both simulated and real sources. It enables researchers and engineers to generate synthetic power system data, analyze real-world measurements, and develop models that can distinguish between authentic and synthetic signals.

## Repository Structure

The repository is organized into three main components:

### Discriminator

The core analysis framework that processes power system data and classifies signals as real or simulated:

- **simprocess**: Core library with feature extraction, machine learning, and analysis capabilities
- **tools**: Utility scripts for data handling
- **main.py**: Primary entry point for the SimProcess framework
- **workflow_example.py**: Example script showing a complete analysis pipeline

### Generators

This directory contains different implementations of power grid simulators:

- **Mosaik**: Generator based on the Mosaik co-simulation framework
- **Pandapower**: Generator based on the Pandapower network calculation framework
- **VariationalRecurrentNeuralNetwork**: VRAE-based generator for power system data

### EPIC

Tools for working with the Electric Power and Intelligent Control (EPIC) dataset:

- **preprocessor.py**: Utility for transforming EPIC dataset measurements to the standardized format used by SimProcess
- Additional support files for EPIC data integration

## Key Features

- **Synthetic Data Generation**: Multiple approaches to generate realistic power grid data with configurable noise profiles
- **Multi-level Noise Modeling**: Support for layered noise types including Gaussian, uniform, Laplace, impulse, and more
- **Feature Extraction**: Statistical feature extraction from time series data
- **Machine Learning Classification**: Models to discriminate between real and simulated signals
- **Visualization Tools**: Comprehensive plotting and analysis utilities
- **Pipeline Integration**: End-to-end workflow from data generation/collection to final classification

## Dataset Sources

The framework works with both synthetic and real-world datasets:

- **Synthetic Data**: Generated using the Mosaik and Pandapower simulators
- **Real Data**: EPIC dataset from the Singapore University of Technology and Design (SUTD)

The EPIC dataset can be requested from iTrust, Centre for Research in Cyber Security at SUTD: [https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

## Getting Started

The repository includes several README files with detailed instructions for each component:

- See Discriminator/README.md for SimProcess analysis framework documentation
- See Generators/Mosaik/README.md and Generators/Pandapower/README.md for data generation tools
- See EPIC/README.md for information on processing real-world power grid data

## License

MIT License
