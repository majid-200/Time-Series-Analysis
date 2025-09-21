# Time Series Analysis and Forecasting in Python 📈

This repository contains a collection of Jupyter notebooks dedicated to exploring and applying various time series analysis and forecasting techniques. The projects are based on guided tutorials from different sources, providing a practical and hands-on approach to learning.

The main objectives of this repository are:
- To implement fundamental and advanced forecasting models from scratch and using specialized libraries.
- To understand the theoretical concepts behind time series components like stationarity, seasonality, and autocorrelation.
- To practice a complete data science workflow: data preprocessing, exploratory analysis, model building, robust evaluation, and visualization.

## 🗂️ Repository Structure

The projects are organized into two main directories based on their source:

-   `Time Series Forcasting Bakery Data/`: A comprehensive, in-depth tutorial covering a wide range of modern forecasting techniques using the `statsforecast` library.
-   `Time Series Forcasting in Python/`: A series of notebooks based on the "Time Series Forecasting in Python" book, focusing on building foundational concepts step-by-step.

---

## 🛠️ Prerequisites & Setup

To run these notebooks, you'll need Python 3 and a few key data science libraries. It is highly recommended to create a virtual environment to manage dependencies.

**Key Libraries:**
- `numpy`
- `pandas`
- `matplotlib`
- `statsmodels`
- `scikit-learn`
- `statsforecast`
- `utilsforecast`

You can install them using pip:
```bash
pip install numpy pandas matplotlib statsmodels scikit-learn statsforecast
```

After cloning the repository, navigate to the project directory and launch Jupyter Lab or Jupyter Notebook:
```bash
git clone https://github.com/majid-200/Time-Series-Analysis.git
cd Time-Series-Analysis
jupyter lab
```

---

## 📺 Time Series Forecasting: French Bakery Sales

This project follows a detailed video tutorial (Time Series Forecasting in Python – Tutorial for Beginners https://www.youtube.com/watch?v=fxx_E0ojKrc), forecasting daily sales for a French bakery. It showcases a complete, modern workflow from data cleaning to advanced model evaluation using the high-performance `statsforecast` library.

### **Notebook: `daily_sales_forecasting.ipynb`**

This single notebook covers an end-to-end forecasting task.

**Analysis & Key Steps:**

1.  **Initial Setup & EDA:** The notebook begins by loading the `daily_sales_french_bakery.csv` dataset. Initial data exploration and visualization are performed for key products like 'BAGUETTE' and 'CROISSANT' to understand their sales patterns over time.

2.  **Data Preprocessing:** The dataset is filtered to only include products with sufficient historical data (at least 28 days), ensuring that the models have enough information to learn from.

3.  **Baseline Models:** A crucial first step in any forecasting project is to establish a baseline. Four simple yet powerful baseline models are implemented and evaluated:
    *   **Naive Forecast:** Predicts the next value will be the same as the last observed value.
    *   **Seasonal Naive:** Predicts the value based on the same time in the previous season (e.g., same day last week).
    *   **Historic Average:** Predicts the average of all past observations.
    *   **Window Average:** Predicts the average of a recent window of observations (e.g., the last 7 days).

4.  **AutoARIMA Modeling:** The notebook moves to a more sophisticated model, `AutoARIMA`, which automatically selects the optimal parameters (p, d, q) for an ARIMA model. Both a standard `ARIMA` and a `Seasonal ARIMA (SARIMA)` are trained and evaluated to capture weekly seasonality.

5.  **Robust Evaluation with Cross-Validation:** Instead of a single train-test split, time series cross-validation is used. This technique creates multiple "windows" of training and testing data, providing a much more reliable estimate of a model's real-world performance.

6.  **Forecasting with Exogenous Features (SARIMAX):** The analysis is enhanced by incorporating an external variable—the `unit_price` of the product. This turns the SARIMA model into a SARIMAX model, testing whether price information can improve sales forecasts.

7.  **Advanced Feature Engineering:** To further improve accuracy, new features are engineered directly from the timestamp:
    *   **Fourier Features:** Trigonometric functions (sine/cosine) are used to model complex seasonal patterns more effectively than standard seasonal dummies.
    *   **Time-Based Features:** Simple calendar features like `day_of_week`, `month`, and `week_of_year` are created.

8.  **Prediction Intervals:** The project demonstrates how to generate prediction intervals, which provide a range of likely outcomes instead of a single point forecast. This is critical for understanding forecast uncertainty and making better business decisions.

9.  **Comprehensive Metric Evaluation:** Finally, all models are compared using a suite of evaluation metrics, including **MAE**, **MAPE**, **RMSE**, and **Scaled CRPS** (for probabilistic forecasts), to determine the best-performing model.

---

## 📚 Book: "Time Series Forecasting in Python" by Marco Peixeiro

This section contains notebooks that follow the chapters of the book, building concepts from the ground up.

### **Notebook 01: Baseline Forecasting**

-   **Concept:** Establishes the importance of baseline models as a benchmark for any serious forecasting task.
-   **Dataset:** Johnson & Johnson quarterly earnings per share (EPS).
-   **Analysis & Key Steps:**
    1.  The notebook begins by plotting the J&J EPS data, showing a clear upward trend and seasonality.
    2.  The data is split into a training and a testing set.
    3.  Four baseline methods are implemented and compared:
        -   Historical Mean
        -   Last Year's Mean
        -   Last Known Value
        -   Naive Seasonal Forecast
    4.  The Mean Absolute Percentage Error (MAPE) is used to evaluate the performance of each baseline. The Naive Seasonal forecast proves to be the most effective, highlighting the strong seasonal component in the data.

### **Notebook 02: Stationarity and Random Walks**

-   **Concept:** Introduces the fundamental concept of **stationarity**—the idea that a time series's statistical properties (like mean and variance) do not change over time. This is a key assumption for many forecasting models.
-   **Dataset:** Google (GOOGL) daily closing stock prices and simulated data.
-   **Analysis & Key Steps:**
    1.  The notebook first simulates both a stationary and a non-stationary (random walk) process to build intuition.
    2.  It visually demonstrates that for a stationary series, the mean and variance are constant, while for a random walk, they are not.
    3.  Introduces two critical tools for identifying non-stationarity:
        -   **Augmented Dickey-Fuller (ADF) Test:** A statistical test where the null hypothesis is that the series is non-stationary. A low p-value allows us to reject this.
        -   **Autocorrelation Function (ACF) Plot:** A visual plot that shows how correlated a data point is with its past values. For a random walk, the ACF shows a very slow, linear decay.
    4.  These tools are applied to the GOOGL stock price data, confirming it behaves like a random walk.
    5.  It concludes by showing that since random walks are unpredictable, only simple naive forecasts are appropriate.

### **Notebook 03: Moving Average (MA) Models**

-   **Concept:** Explores the Moving Average (MA) model, which forecasts the next value based on the residual errors from previous forecasts.
-   **Dataset:** Monthly widget sales.
-   **Analysis & Key Steps:**
    1.  The widget sales data is first checked for stationarity and is found to be non-stationary.
    2.  **Differencing** is applied (subtracting the previous value from the current value) to make the series stationary.
    3.  The **ACF plot** is used on the *differenced* data. A key characteristic of an MA(q) process is that its ACF plot shows significant correlations up to lag `q` and then abruptly cuts off. Here, the plot cuts off after lag 2, identifying the process as **MA(2)**.
    4.  A rolling forecast is implemented using a `SARIMAX` model configured to be an MA(2) model.
    5.  The MA(2) model's performance is compared against baseline models using Mean Squared Error (MSE), demonstrating its superior accuracy.
    6.  Finally, the forecasted differenced values are transformed back to the original scale to get the final sales predictions.
