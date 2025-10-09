# Code for Joint Energy-based Models (JEM)

This directory contains the Python code for implementing and training a Joint Energy-based Model (JEM).

## Files

- `data.py`: This script handles the loading and preprocessing of the dataset used for training and evaluation. It is responsible for preparing the data in a suitable format for the model.
- `JEM.ipynb`: A Jupyter Notebook that provides an interactive environment for experimenting with the JEM model, visualizing results, and demonstrating key concepts. It serves as a practical guide and a space for exploration.
- `loss.py`: Defines the loss functions used for training the JEM model. This is a critical component for optimizing the model's parameters during the training process.
- `models.py`: Contains the Python classes and functions that define the architecture of the Joint Energy-based Model. This is the core of the implementation, where the model's structure is specified.
- `sampling.py`: Implements methods for generating samples from the trained JEM model. This is used to produce new data points that follow the distribution learned by the model.
- `train.py`: The main script for training the JEM model. It orchestrates the data loading, model definition, and training loop, bringing all the other components together to train the model.
