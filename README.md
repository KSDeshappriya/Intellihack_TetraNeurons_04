# Stock Price Prediction Project

## 1. Project Overview

This project aims to predict the **5-day future closing price** of a stock using a combination of traditional machine learning models and deep learning methods. The dataset consists of historical stock prices, and the project applies advanced feature engineering and model evaluation techniques.

## 2. Key Objectives

- Perform **Exploratory Data Analysis (EDA)** to identify trends and relationships.
- Engineer **technical indicators** to enhance predictive features.
- Evaluate multiple models including **Linear Regression**, **Random Forest**, **Gradient Boosting**, and **LSTM**.
- Select the best model based on **RMSE**, **MAE**, and **Directional Accuracy**.
- Simulate trading strategies to assess model effectiveness.

## 3. Key Findings

- **Gradient Boosting Regressor** achieved the best performance in the trading simulation:
    - **Final Portfolio Value**: $17,564.66
    - **Total Return**: 75.65%

- Model performance summary:

| Model               | RMSE     | MAE      | Directional Accuracy |
|---------------------|----------|----------|----------------------|
| Linear Regression   | 5.0490   | 3.5347   | N/A                  |
| Random Forest       | 28.3768  | 16.6875  | N/A                  |
| Gradient Boosting   | 28.9442  | 16.9419  | N/A                  |
| LSTM                | 147.9759 | 144.6760 | 47.83%               |

- **Feature Engineering** enhanced model accuracy by incorporating technical indicators like moving averages (MA), MACD, RSI, and Bollinger Bands.

## 4. Project Structure

```
├── notebooks
│   └── Intellihack_TetraNeurons_04.ipynb   # Main code file for the entire pipeline
├── models
│   ├── gradient_boosting_model.pkl         # Saved Gradient Boosting model
│   └── lstm_model.h5                       # Saved LSTM model
├── data
│   └── question4-stock-data.csv               # data
├── docs
│   ├── odt/
│   ├── EDA-Report.pdf
│   ├── End-to-End_System_Design-Documentation.pdf
│   └── Model-Selection-Documentation.pdf
├── plots
│   ├── closing_prices.png                  # Historical closing price plot
│   ├── volume.png                          # Trading volume plot
│   ├── correlation.png                     # Correlation matrix plot
│   ├── seasonal_decomp.png                 # Seasonal decomposition plot
│   └── feature_importance.png              # Gradient Boosting feature importance
├── stock_predictions.csv               # Final predictions
```

## 5. Dependencies

Ensure the following Python packages are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow statsmodels joblib
```

### Required Libraries:

- Python 3.8+
- TensorFlow (for LSTM implementation)
- scikit-learn (for traditional ML models)
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- statsmodels (seasonal decomposition)

## 6. Running the Project

1. **Set up your environment:**

Ensure all dependencies are installed using the command above.

2. **Execute:**

```bash
notebooks/Intellihack_TetraNeurons_04.ipynb
```

3. **Check outputs:**

- Model artifacts: `models/gradient_boosting_model.pkl`, `models/lstm_model.h5`
- Predictions: `stock_predictions.csv`
- Visualizations: Closing prices, volume, correlation matrix, seasonal decomposition, and feature importance plots

## 7. Reproducing Results

To reproduce the results:

1. Ensure the dataset `question4-stock-data.csv` is in the working directory.
2. Execute the script as described above.
3. Analyze the saved outputs for model performance and trading outcomes.

## 8. Future Improvements

- Implement **Hybrid Models**: Combine Gradient Boosting and LSTM for improved performance.
- Perform **Hyperparameter Tuning**: Optimize model parameters using GridSearch or RandomizedSearch.
- Incorporate **External Factors**: Include news sentiment, macroeconomic indicators, or other external drivers.