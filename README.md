# Heart Disease Prediction using Supervised Learning

## Overview
This project applies supervised machine learning techniques to predict heart disease based on various health-related attributes. The primary models used in this notebook are:
- **Logistic Regression**
- **Random Forest Classifier**

## Dataset
The dataset used is `heart_2020_cleaned.csv`, which contains various health indicators relevant to heart disease prediction.

## Requirements
To run this notebook, you need the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Steps in the Notebook
1. **Importing Libraries**: Essential libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn are imported.
2. **Data Loading**: The dataset is loaded using Pandas.
3. **Exploratory Data Analysis (EDA)**: Visualizations and statistical summaries to understand the dataset.
4. **Feature Engineering & Preprocessing**: Data cleaning, encoding categorical variables, and handling missing values.
5. **Model Training**:
   - Logistic Regression
   - Random Forest Classifier
6. **Hyperparameter Tuning**: GridSearchCV is used to optimize model performance.
7. **Evaluation**: Models are assessed using accuracy, precision, recall, F1-score, and confusion matrix.

## How to Run
1. Download the dataset `heart_2020_cleaned.csv` and place it in the working directory.
2. Open and run `Final Task.ipynb` in Jupyter Notebook or Google Colab.
3. Follow the steps outlined in the notebook to train and evaluate the models.

## Results
The notebook provides visualizations and performance metrics to compare model effectiveness. The best-performing model can be selected based on evaluation results.

## Acknowledgments
This project utilizes data science techniques to analyze heart disease risk factors and predict potential cases using machine learning.

