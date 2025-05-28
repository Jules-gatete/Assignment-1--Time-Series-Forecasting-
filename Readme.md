# PM2.5 Time Series Forecasting

## Overview

Predict hourly PM2.5 concentrations using a Long Short-Term Memory (LSTM) model tuned with Optuna. This project was developed for the Kaggle competition **Assignment 1 - Time Series Forecasting (May 2025)** and achieved a public RMSE of **4740.73**. and (**Validation Performance Metrics: RMSE: 67.09, MAE: 45.18 and R² Score: 0.27**)

## Repository Structure

```
├── Assignment_1_Time_Series_Forecasting.ipynb   # Data cleaning, feature engineering, model training, tuning, and visualization
├── submission3.csv         # Best model predictions (public RMSE: 4740.7316)
├── report.pdf              # Detailed methodology, experiments, and results
└── README.md               # Project overview and instructions
```

## Methodology

1. **Data Preprocessing**

   * Loaded 30,676 training and 13,148 test records
   * Parsed `datetime` and extracted cyclical time features (hour, month) and categorical flags (weekend)
   * Interpolated missing PM2.5 values and forward/backward filled other gaps
   * Scaled inputs and target with MinMaxScaler
   * Created overlapping sequences (length=36, stride=2) for LSTM input

2. **Model Design**

   * Single-layer bidirectional LSTM (64 units) + Dropout (0.2)
   * Dense layer (16 units, ReLU) → output
   * Mixed precision, Adam optimizer (lr=0.000641), MSE loss
   * Early stopping and learning rate reduction to prevent overfitting

3. **Hyperparameter Tuning**

   * Used Optuna to search over:

     * Sequence length, units, number of layers, dropout, learning rate, batch size, stride
   * **Best trial**: `seq_length=36`, `units=64`, `layers=1`, `dropout=0.2`, `lr=0.000641`, `batch=64`, `stride=2` → validation RMSE **76.94**

4. **Results**

   * Public RMSE (kaggle): **4740.73** (submission3.csv) & 
   * Validation Performance Metrics:
   * RMSE: 67.09
   * MAE: 45.18
   * R² Score: 0.27
   * Validation (last 100 samples):**RMSE=67.09, MAE=45.18, R²≈0.27**
   * Key predictors: temperature, dew point, and time-of-day patterns

## Usage

1. **Install dependencies**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow optuna
   ```

2. **Run the notebook**

   * Open `pm25_prediction.ipynb` in Colab or Jupyter
   * Place `train.csv`, `test.csv`, and `sample_submission.csv` in the working directory
   * Execute all cells to produce `submission3.csv` and visualization outputs

3. **Review the report**

   * See `report.pdf` for in-depth analysis, figures, and next steps

## Next Steps & Improvements

* Incorporate additional lag features (e.g., traffic data, meteorological forecasts)
* Experiment with attention mechanisms or transformer-based models
* Increase Optuna trials for finer hyperparameter search
* Analyze residuals and outliers for targeted model refinement

---

**Kaggle Submission History**

* submission3.csv: RMSE 4740.73 (Best)
* submission1.csv: RMSE 5659.54
* subm\_fixed.csv: RMSE 5868.05
* submissiontest.csv: RMSE 6369.64

*Note: failed submissions had formatting or row-count issues.*
