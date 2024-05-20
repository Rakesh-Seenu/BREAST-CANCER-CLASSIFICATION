# Breast Cancer Classification

This project involves predicting whether a breast cancer diagnosis is benign or malignant using machine learning techniques, specifically a Support Vector Machine (SVM). The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which is included in the scikit-learn library.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to build a classifier that can accurately predict if a breast cancer tumor is benign or malignant based on various features of cell nuclei present in the digitized image of a fine needle aspirate (FNA) of a breast mass.

## Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset contains 569 instances of tumors, each with 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes the following:

- 569 samples
- 30 numerical features
- 1 target label (0 for benign, 1 for malignant)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

''' pip install pandas numpy matplotlib seaborn scikit-learn '''

## Usage

1. Clone the repository:

git clone https://github.com/your-username/breast-cancer-classification.git
cd breast-cancer-classification

2. Run the Jupyter Notebook or Python script to train and evaluate the model. Ensure you have Jupyter installed if you want to run the notebook:

jupyter notebook Breast_Cancer_Classification.ipynb

## Results
The Support Vector Machine (SVM) classifier, after hyperparameter tuning using grid search, achieved an accuracy of 97% in classifying breast tumors as benign or malignant. Below are the steps involved in the process:

1. Data Loading and Preprocessing: Loaded the dataset and created a DataFrame for easy manipulation.
2. Data Visualization: Visualized the data using pair plots, count plots, and heatmaps to understand feature relationships.
3. Model Training: Trained an initial SVM model.
4. Feature Scaling: Improved model performance by scaling the features.
5. Hyperparameter Tuning: Used GridSearchCV to find the best hyperparameters for the SVM model.
6. Evaluation: Evaluated the model using a confusion matrix and classification report.
