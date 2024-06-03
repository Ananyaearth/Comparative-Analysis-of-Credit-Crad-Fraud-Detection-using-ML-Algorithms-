# Credit Card Fraud Detection

This repository contains a comprehensive analysis and implementation of various machine learning models for credit card fraud detection using the Credit Card Fraud Detection dataset. The dataset contains transactions made by credit cards in September 2013 by European cardholders.

## Introduction

Credit card fraud is a significant problem in the financial sector, causing substantial losses annually. Detecting fraudulent transactions is a critical task that requires efficient and accurate algorithms. This project aims to explore and compare different machine learning models to identify fraudulent transactions.

## Dataset

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains 284,807 transactions, with 492 frauds, representing 0.172% of all transactions. The dataset is highly imbalanced, with the majority of transactions being non-fraudulent.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Ananyaearth/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Exploratory Data Analysis (EDA)

Before building the models, we performed exploratory data analysis to understand the distribution of the data, the class imbalance, and the correlation between features. Key steps include:

- Viewing the first few rows of the dataset.
- Checking for missing values.
- Plotting the class distribution.
- Visualizing the distribution of features using histograms.
- Generating a correlation matrix to identify correlations between features.

## Modeling

We implemented and compared several machine learning models and ensemble techniques, including:

1. **Ensemble Models:**
   - Bagging with Decision Trees
   - Averaging Method
   - Max Voting
   - Stacking
   - Blending

2. **Boosting:**
   - AdaBoost

3. **Artificial Neural Networks (ANN):**
   - Multi-layer Perceptron

4. **Traditional Machine Learning Models:**
   - Naive Bayes
   - K-Nearest Neighbors (KNN)
   - Linear Discriminant Analysis (LDA)
   - Decision Tree
   - Support Vector Machine (SVM)
   - Logistic Regression
   - Random Forest

## Evaluation

The models were evaluated using several metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics were chosen to provide a comprehensive evaluation of the models, particularly given the imbalanced nature of the dataset.

## Results

The results of the models are summarized in the table below:

| Algorithm                   | Accuracy   | Precision | Recall    | F1 Score  |
|-----------------------------|------------|-----------|-----------|-----------|
| Artificial Neural Network   | 0.9994     | 0.8478    | 0.7959    | 0.8211    |
| Naive Bayes                 | 0.9930     | 0.1462    | 0.6327    | 0.2375    |
| K-Nearest Neighbors         | 0.9984     | 1.0000    | 0.0510    | 0.0971    |
| Linear Discriminant Analysis| 0.9994     | 0.8690    | 0.7449    | 0.8022    |
| Decision Tree               | 0.9991     | 0.6964    | 0.7959    | 0.7429    |
| Logistic Regression         | 0.9683     | 0.9895    | 0.9467    | 0.9676    |
| Random Forest               | 0.9995     | 0.9302    | 0.7921    | 0.8556    |
| Support Vector Machine      | 0.9993     | 0.8141    | 0.7938    | 0.8038    |
|                             |            |           |           |           |
| **Ensemble Models**         |            |           |           |           |
| Averaging Method            | 0.9996     | 0.9333    | 1.0000    | 0.9655    |
| Max Voting                  | 0.9992     | 1.0000    | 0.8571    | 0.9231    |
| Stacking                    | 0.9982     | 1.0000    | 0.9967    | 1.0000    |
| Blending                    | 0.9987     | 1.0000    | 0.7857    | 0.8800    |
| Bagging                     | 0.9983     | 0.9167    | 0.7857    | 0.8462    |
| Boosting                    | 0.9992     | 1.0000    | 0.8571    | 0.9231    |

## Conclusion

Through this project, we implemented and compared various machine learning models to detect credit card fraud. The ensemble models, particularly the Stacking and Averaging methods, provided the best performance in terms of accuracy, precision, recall, and F1 score. This demonstrates the potential of ensemble techniques in handling imbalanced datasets and improving prediction performance.
