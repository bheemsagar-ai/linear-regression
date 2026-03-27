# Linear Regression - Height Prediction

A simple linear regression model that predicts **Height** based on **Weight** using scikit-learn.

## Project Structure

```
linear_regression/
├── linear.py       # Main script
├── height.csv      # Dataset
└── README.md
```

## Workflow

1. Load and explore the dataset (`height.csv`)
2. Split data into train/test sets (80/20)
3. Standardize features using `StandardScaler`
4. Train a `LinearRegression` model
5. Evaluate using MSE, MAE, RMSE, R², and Adjusted R²

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

```bash
python linear.py
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| Adjusted R² | R² adjusted for number of features |
