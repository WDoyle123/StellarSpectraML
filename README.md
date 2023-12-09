# StellarSpectraML

## Contents
- [Overview](#overview)
- [Technical Features](#technical-features)
- [TensorFlow and Neural Network Architecture](#tensorflow-and-neural-network-architecture)
- [Files Description](#files-description)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Output](#output)
- [Acknowledgements](#acknowledgements)

## Overview
This project aims to classify stellar objects based on their spectral types using machine learning techniques. It leverages data from the Yale Bright Star Catalogue and the SIMBAD astronomical database to train a neural network model.

## Technical Features

- **TensorFlow for Neural Networks**: Utilises TensorFlow for creating and training the neural network model.
- **Data Handling with Pandas**: Employs Pandas for efficient data manipulation and analysis.
- **Scikit-Learn for Data Preprocessing**: Incorporates Scikit-Learn for data preprocessing tasks like train-test split and feature scaling.
- **Seaborn and Matplotlib for Data Visualisation**: Uses Seaborn and Matplotlib for visualising data trends and model performance metrics.
- **Label Encoding**: Implements label encoding for categorical data conversion, facilitating the machine learning process.
- **Model Optimisation and Evaluation**: Utilises Adam optimiser for efficient training and evaluates the model using accuracy metrics.
- **Dropout for Regularisation**: Integrates dropout layers in the neural network to prevent overfitting.
- **Custom Data Handler Script**: Includes a `data_handler.py` script for specific data retrieval, cleaning, and preprocessing tailored to astronomical data.

### TensorFlow and Neural Network Architecture
- This project utilsed TensorFlow for to create a neural network.
- **Model Architecture**: 
  - **Input Layer**: Receives the scaled features (visual magnitude and B-V color index).
  - **Hidden Layers**: Two dense layers with 64 and 32 neurons respectively, using ReLU (Rectified Linear Unit) activation function.
  - **Dropout Layers**: With a rate of 0.1 after each dense layer to prevent overfitting.
  - **Output Layer**: A dense layer with 10 neurons (corresponding to the number of spectral classes), using the Softmax activation function for multi-class classification.
- **Optimisation and Loss Function**: 
  - **Optimiser**: Adam optimiser with a learning rate of 0.001. 
  - **Loss Function**: Sparse Categorical Crossentropy, suitable for multi-class classification problems where each class is mutually exclusive.

## Files Description
- `data_handler.py`: This script handles data retrieval and preprocessing. It includes functions for reading data files, querying the SIMBAD database for spectral types, and combining data into a single DataFrame.
- `main.py`: The main script where the machine learning model is defined, trained, and evaluated. It includes data preprocessing steps, model creation, and performance plotting.

## Usage

### Data Preparation
Run `data_handler.py` to create a file to query the SIMBAD database and prepare the combined dataset.

```bash
python data_handler.py
```

### Model Training and Evaluation
Execute `main.py` to train the machine learning model on the prepared data. This script splits the data into training and testing sets, scales the features, trains a neural network model, and evaluates its performance.

```bash
python main.py
```

## Output
After running `main.py`, the script will output the model's accuracy and save a plot of the training and validation accuracy over epochs as `model_accuracy_plot.png`.

## Acknowledgements

[Yale Bright Star Catalogue](https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name%3Dbsc5p&Action=More+Options)

[SIMBAD](http://simbad.cds.unistra.fr/simbad/sim-fscript)

## [Back to Top](#StellarSpectraML)