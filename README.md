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
- `tensorflow`
- `torch`
You can install them using pip:
```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn statsforecast utilsforecast tensorflow torch
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

### **Notebook 04: Autoregressive (AR) Models**

-   **Concept:** Focuses on the Autoregressive (AR) model, where the forecast for the next value is a linear combination of past observed values. It introduces the **Partial Autocorrelation Function (PACF)** as the primary tool for identifying the order of an AR model.
-   **Dataset:** Weekly foot traffic data.
-   **Analysis & Key Steps:**
    1.  The foot traffic data is loaded and visualized, revealing a non-stationary pattern. This is confirmed with an Augmented Dickey-Fuller (ADF) test.
    2.  First-order differencing is applied to the series, which successfully transforms it into a stationary series (confirmed by a second ADF test).
    3.  The notebook introduces the key difference between ACF and PACF plots for model identification:
        -   An **ACF plot** of an AR(p) process will show a slow, geometric decay.
        -   A **PACF plot** of an AR(p) process will show significant correlations up to lag `p` and then abruptly cut off.
    4.  By plotting the PACF of the stationary (differenced) foot traffic data, a clear cutoff is observed after lag 3. This indicates that an **AR(3)** model is the appropriate choice.
    5.  A rolling forecast is implemented to compare the performance of the AR(3) model against two baselines: a historical mean forecast and a naive (last value) forecast.
    6.  The models are evaluated using Mean Squared Error (MSE), where the AR(3) model significantly outperforms the baselines on the differenced data.
    7.  Finally, the predictions from the AR(3) model are inverse-transformed to revert them back to the original scale of the foot traffic data, providing a final, interpretable forecast.

### **Notebook 05: Autoregressive Moving Average (ARMA) Models**

-   **Concept:** Combines the AR and MA models into the more general **ARMA(p,q) model**. This notebook introduces a complete and robust modeling procedure, as visual identification with ACF/PACF plots is no longer effective for ARMA processes. The focus shifts to programmatic model selection using the **Akaike Information Criterion (AIC)** and performing rigorous **residual analysis** to validate the model's performance.
-   **Dataset:** Hourly bandwidth usage data.
-   **Analysis & Key Steps:**
    1.  The notebook begins by demonstrating that for a stationary ARMA process, both the ACF and PACF plots exhibit a slow, decaying pattern, making it difficult to determine the model orders (`p` and `q`) visually.
    2.  It introduces the **Akaike Information Criterion (AIC)** as the primary method for model selection. A function is created to perform a grid search, fitting multiple ARMA(p,q) models and selecting the combination with the lowest AIC value.
    3.  A critical new step, **residual analysis**, is introduced to ensure the chosen model is a good fit. The residuals (the model's errors) must resemble white noise (i.e., be uncorrelated and normally distributed). Two methods are used for this validation:
        -   **Q-Q Plot:** A visual tool to check if the residuals follow a normal distribution.
        -   **Ljung-Box Test:** A statistical test to confirm that the residuals are independent and not correlated over time.
    4.  This complete, general procedure is then applied to the hourly bandwidth dataset. The data is first made stationary through differencing.
    5.  The AIC-based grid search is run on the stationary data, identifying an **ARMA(2,2)** model as the optimal choice.
    6.  The chosen model's residuals are thoroughly analyzed using diagnostic plots and the Ljung-Box test. The results confirm that the residuals behave like white noise, validating the model's fit.
    7.  Finally, a rolling forecast is implemented. The ARMA(2,2) model's predictions significantly outperform baseline models, and the forecasted values are inverse-transformed back to the original bandwidth usage scale.

### **Notebook 06: Autoregressive Integrated Moving Average (ARIMA) Models**

-   **Concept:** Introduces the **Autoregressive Integrated Moving Average (ARIMA)** model, which extends the ARMA model to handle non-stationary time series. The "I" in ARIMA stands for "Integrated" and represents the differencing step required to make the data stationary before applying the ARMA model.
-   **Dataset:** Johnson & Johnson quarterly earnings per share (EPS).
-   **Analysis & Key Steps:**
    1.  The J&J EPS data is loaded and confirmed to be non-stationary using the Augmented Dickey-Fuller (ADF) test.
    2.  The notebook demonstrates how to determine the order of integration (`d`). First-order differencing is applied, but the series remains non-stationary. After a second round of differencing, the series becomes stationary, establishing that the correct order is **`d=2`**.
    3.  With `d` determined, the general modeling procedure from the previous notebook is applied. A grid search is performed over various `p` and `q` values to find the model with the lowest **Akaike Information Criterion (AIC)**.
    4.  The grid search identifies an **ARIMA(3,2,3)** model as the best fit for the data.
    5.  A comprehensive residual analysis is performed on the ARIMA(3,2,3) model. Diagnostic plots and the Ljung-Box test confirm that the residuals are uncorrelated and normally distributed, validating the model's quality.
    6.  Finally, the model is used to forecast the last four quarters of data. Its performance is compared against a naive seasonal baseline using the **Mean Absolute Percentage Error (MAPE)**.
    7.  The ARIMA model achieves a remarkably low MAPE of 1.71%, significantly outperforming the baseline's 11.56%, showcasing its effectiveness in capturing both trend and seasonality in the data.

### **Notebook 07: Seasonal ARIMA (SARIMA) Models**

-   **Concept:** Extends the ARIMA framework to model time series with a clear seasonal component. This notebook introduces the **Seasonal ARIMA (SARIMA)** model, which incorporates seasonal parameters to capture repeating patterns over time. It also utilizes **time series decomposition** as a primary tool for identifying and understanding seasonality.
-   **Dataset:** The classic "Air Passengers" dataset, which exhibits both a strong upward trend and distinct yearly seasonality.
-   **Analysis & Key Steps:**
    1.  **Time Series Decomposition:** The notebook begins by applying **Seasonal-Trend decomposition using LOESS (STL)** to the air passenger data. This powerful technique separates the time series into three components: **trend**, **seasonality**, and **residuals**, visually confirming the strong yearly pattern in the data.
    2.  **Seasonal Differencing:** To handle seasonality, the concept of **seasonal differencing** is introduced. This involves subtracting the value from the previous season (e.g., the value from 12 months ago) to remove the repeating pattern and help make the series stationary. The data is made stationary by applying both a first-order regular difference (`d=1`) and a first-order seasonal difference (`D=1`).
    3.  **SARIMA Model Introduction:** The notebook formally defines the **SARIMA(p,d,q)(P,D,Q)m** model. This model includes the standard ARIMA parameters (`p,d,q`) plus a new set of seasonal parameters:
        -   `P`: Order of the seasonal autoregressive (AR) component.
        -   `D`: Order of seasonal integration (differencing).
        -   `Q`: Order of the seasonal moving average (MA) component.
        -   `m`: The number of time steps in a single seasonal period (e.g., `m=12` for monthly data with a yearly cycle).
    4.  **Systematic Model Selection:** A comprehensive grid search is performed to find the optimal combination of all seven SARIMA parameters. The function iterates through numerous combinations and selects the model with the lowest **Akaike Information Criterion (AIC)**.
    5.  **Residual Analysis and Validation:** The best-performing model, a **SARIMA(2,1,1)(1,1,2,12)**, is selected. A thorough residual analysis is conducted using diagnostic plots and the Ljung-Box test. The results confirm that the residuals are uncorrelated and normally distributed, validating that the model has successfully captured the underlying patterns.
    6.  **Forecasting and Evaluation:** The SARIMA model is used to forecast future air passenger numbers. Its performance is compared against both a standard (non-seasonal) ARIMA model and a naive seasonal baseline. The SARIMA model achieves the lowest **Mean Absolute Percentage Error (MAPE)** of 2.85%, demonstrating its superior ability to handle seasonal data.
 
### **Notebook 08: SARIMA with Exogenous Variables (SARIMAX)**

-   **Concept:** Introduces the most general and powerful model in the ARIMA family: **SARIMAX**. The "X" stands for "eXogenous," meaning the model can incorporate external predictor variables to improve the forecast of the target variable. This is particularly useful in economics and business, where factors like consumption, investment, and inflation can influence outcomes like GDP.
-   **Dataset:** US Macroeconomic data from 1959 to 2009, including the target variable `realgdp` (Real Gross Domestic Product) and several potential exogenous predictors like `realcons` (real consumption) and `realinv` (real investment).
-   **Analysis & Key Steps:**
    1.  **Exploring Exogenous Variables:** The notebook begins by loading the US macroeconomic dataset and visualizing the target variable (`realgdp`) alongside potential predictors. The goal is to see if other economic indicators move in a way that might help predict GDP.
    2.  **Model Building with SARIMAX:** The general modeling procedure is adapted for SARIMAX. The target variable (`realgdp`) is made stationary through first-order differencing (`d=1`). A grid search is then performed to find the optimal SARIMA parameters, but this time, the `exog` (exogenous variables) parameter is included in the model fitting.
    3.  **Model Selection and Validation:** The grid search identifies a **SARIMAX(3,1,3)(0,0,0,4)** model as the best fit based on the **Akaike Information Criterion (AIC)**. A rigorous residual analysis confirms that the model's errors are uncorrelated and normally distributed, validating its statistical soundness. The summary also shows that variables like `realcons` and `realinv` have statistically significant coefficients, confirming their predictive power.
    4.  **Recursive Forecasting:** A crucial concept for SARIMAX is introduced: **recursive forecasting**. Since the model requires future values of the exogenous variables to make a forecast (which are often unknown), a common strategy is to make one-step-ahead forecasts. A recursive function is built to:
        -   Train the model on an initial window of data.
        -   Forecast the next time step.
        -   Add the new true data to the training set.
        -   Re-fit the model and repeat the process.
    5.  **Performance Evaluation:** The recursive SARIMAX forecast is compared to a naive (last value) baseline. The SARIMAX model achieves a slightly better **Mean Absolute Percentage Error (MAPE)** of 0.70% compared to the baseline's 0.74%. This demonstrates that even with a strong trend where a naive forecast is effective, incorporating relevant external data can provide a measurable improvement in accuracy.

### **Notebook 09: Vector Autoregressive (VAR) Models**

-   **Concept:** Shifts from univariate to **multivariate time series forecasting** with the **Vector Autoregressive (VAR)** model. Unlike previous models that forecast a single series, VAR models capture the interdependencies between multiple time series and forecast them simultaneously. A key assumption is that each variable not only depends on its own past values but also on the past values of the other variables in the system.
-   **Dataset:** US Macroeconomic data, focusing on two interdependent variables: `realdpi` (real disposable personal income) and `realcons` (real personal consumption).
-   **Analysis & Key Steps:**
    1.  **Introducing VAR:** The notebook explains that in a VAR model, each variable is modeled as a linear combination of its own past values and the past values of all other variables in the system. This makes it ideal for analyzing dynamic relationships, like how changes in income might affect consumption, and vice versa.
    2.  **Stationarity and Model Selection:** The first step is to ensure all time series in the system are stationary. Both `realdpi` and `realcons` are non-stationary and require first-order differencing. An AIC-based grid search is then performed to determine the optimal lag order (`p`) for the VAR model, which is found to be `p=3`.
    3.  **Granger Causality Test:** A critical validation step for VAR models is the **Granger causality test**. This statistical test is used to determine if one time series is useful in forecasting another. For a VAR model to be valid, each variable must Granger-cause the others. The test results show p-values less than 0.05, confirming a bidirectional causal relationship between income and consumption, thus validating the use of a VAR(3) model.
    4.  **Model Fitting and Residual Analysis:** The VAR(3) model is fitted to the stationary data. A separate residual analysis is performed for each variable's forecast. The diagnostic plots and Ljung-Box tests confirm that the residuals for both `realdpi` and `realcons` are uncorrelated and normally distributed.
    5.  **Forecasting and Evaluation:** A rolling forecast is performed, and the predictions are inverse-transformed back to their original scales. The VAR(3) model's performance is compared to a naive baseline. The results are mixed:
        -   For `realcons`, the VAR model outperforms the baseline.
        -   For `realdpi`, the VAR model performs worse than the baseline.
    6.  The notebook concludes by introducing the **VARMA** and **VARMAX** models as potential improvements, setting the stage for more complex multivariate analysis.
 
### **Notebook 10: A Complete Forecasting Project**

-   **Concept:** This capstone notebook brings together all the concepts from the previous chapters into a single, end-to-end forecasting project. It simulates a real-world scenario by walking through a structured workflow: data exploration, preprocessing, model selection, robust forecasting, and final evaluation.
-   **Dataset:** Monthly sales of an anti-diabetic drug in Australia from 1991 to 2008.
-   **Analysis & Key Steps:**
    1.  **Exploration and Decomposition:** The project begins by loading and visualizing the data, which clearly shows an upward trend and strong yearly seasonality. **STL decomposition** is used to formally separate and examine the trend, seasonal, and residual components.
    2.  **Achieving Stationarity:** The data is confirmed to be non-stationary. A combination of first-order regular differencing (`d=1`) and first-order seasonal differencing (`D=1`) is applied, successfully transforming the data into a stationary series.
    3.  **Train-Test Split:** A holdout set of the last 36 months (3 years) is created to serve as the test set for final model evaluation.
    4.  **Robust Model Selection:** A comprehensive grid search is performed to find the optimal SARIMA parameters. The search function is enhanced with several validation checks to ensure model stability and reliability, automatically discarding models that fail to converge, have invalid AIC values, or produce extreme parameter estimates. The best model is identified as **SARIMA(2,1,3)(1,1,3,12)**.
    5.  **Residual Analysis:** The chosen model's residuals are meticulously checked using diagnostic plots and the Ljung-Box test. The analysis confirms that the residuals are statistically indistinguishable from white noise, validating the model's fit.
    6.  **Rolling Forecast Implementation:** To simulate a real-world deployment, a **rolling forecast** is implemented. This technique iteratively re-trains the model as new data becomes available, providing a more realistic and robust performance estimate. The SARIMA model is used to forecast the entire 36-month test period one step at a time.
    7.  **Final Evaluation:** The SARIMA model's rolling forecast is compared against a naive seasonal baseline. The results are definitive:
        -   Naive Seasonal MAPE: **12.69%**
        -   SARIMA Model MAPE: **7.90%**
    8.  The significant improvement in accuracy demonstrates the power of the systematic SARIMA modeling procedure in capturing complex trend and seasonal patterns, successfully concluding the foundational part of the book.
 
### **Notebook 11: Introduction to Deep Learning for Time Series Forecasting**

-   **Concept:** This notebook serves as a bridge from classical statistical models to modern deep learning techniques for time series forecasting. It emphasizes the importance of a meticulous and structured data preprocessing workflow, which is a prerequisite for successfully applying complex neural network architectures. The focus is on preparing data for deep learning models rather than building the models themselves.
-   **Dataset:** Metro Interstate Traffic Volume, a large dataset containing hourly traffic data along with weather features.
-   **Analysis & Key Steps:**
    1.  **Data Cleaning and Imputation:** The project starts by addressing common data quality issues.
        -   **Duplicates:** Duplicate timestamps are identified and removed.
        -   **Missing Timestamps:** The notebook identifies gaps in the hourly data. A complete date range is generated, and the original data is merged with it to explicitly show missing rows as `NaN` values.
        -   **Imputation:** Missing values for weather and traffic are imputed by filling them with the average value for that specific hour of the day, preserving the daily cyclical patterns.
    2.  **Feature Engineering:** The notebook demonstrates how to transform raw data into features that are more suitable for machine learning models.
        -   **Feature Selection:** Redundant or uninformative features (`rain_1h`, `snow_1h`) are identified through statistical analysis and removed to simplify the model.
        -   **Cyclical Feature Encoding:** A critical step for time series data is encoding cyclical features like time of day or day of the year. The `date_time` column is converted from a simple timestamp into two new features, `day_sin` and `day_cos`, using sine and cosine transformations. This allows the model to understand the cyclical nature of time (e.g., that 11 PM is close to 1 AM).
    3.  **Data Splitting and Scaling:**
        -   **Train-Validation-Test Split:** The data is split into three distinct sets: a training set (70%), a validation set (20%), and a test set (10%). This three-way split is standard practice in deep learning to allow for hyperparameter tuning on the validation set without touching the final test set.
        -   **Data Scaling:** All features are scaled to a range between 0 and 1 using `MinMaxScaler`. This is a crucial step for neural networks, as it ensures that all features have a similar scale, which helps the model converge faster and perform better. The scaler is fitted *only* on the training data to prevent data leakage from the validation and test sets.
      
### **Notebook 12: Baseline Models for Deep Learning**

-   **Concept:** Lays the foundational groundwork for building deep learning models by establishing various baseline models. This notebook introduces the critical concept of **data windowing**—structuring time series data into input sequences and corresponding labels (targets) that neural networks can process. It implements this logic in a reusable `DataWindow` class and then builds and evaluates several baseline models for different forecasting tasks.
-   **Dataset:** The preprocessed Metro Interstate Traffic Volume dataset from the previous notebook.
-   **Analysis & Key Steps:**
    1.  **Data Windowing:** The core concept of the notebook is creating "windows" of data. A custom `DataWindow` class is built (in both TensorFlow and PyTorch) to handle this process automatically. This class takes the raw time series and transforms it into batches of `(inputs, labels)`, where:
        -   `inputs`: A sequence of past observations (e.g., the last 24 hours of data).
        -   `labels`: The future values to be predicted (e.g., the next hour or the next 24 hours).
    2.  **Single-Step Forecasting Baseline:**
        -   **Task:** Predict the traffic volume for the very next hour.
        -   **Baseline Model:** A simple "last value" model that predicts the next value will be the same as the current one.
        -   **Result:** This naive but effective model sets the performance benchmark that more complex models must beat.
    3.  **Multi-Step Forecasting Baselines:**
        -   **Task:** Predict the traffic volume for the next 24 hours in a single shot.
        -   **Baseline 1 (Last Value):** Predicts that all 24 future values will be the same as the last known value. This performs poorly as it fails to capture any daily patterns.
        -   **Baseline 2 (Repeat):** Predicts that the next 24 hours will be an exact repeat of the previous 24 hours. This is a much stronger baseline as it captures the daily seasonality of the traffic data.
    4.  **Multi-Output Forecasting Baseline:**
        -   **Task:** Simultaneously predict two different variables—`traffic_volume` and `temp`—for the next hour.
        -   **Baseline Model:** A "last value" model adapted to predict multiple targets.
        -   **Result:** This establishes a benchmark for a more complex multivariate forecasting scenario.
    5.  **Implementation in TensorFlow and PyTorch:** A key feature of this notebook is that all concepts—the `DataWindow` class and every baseline model—are implemented in parallel using both **TensorFlow** and **PyTorch**. This provides a practical comparison of the two leading deep learning frameworks and demonstrates how to build equivalent data pipelines and simple models in each.

By the end of the notebook, a clear set of performance metrics (MAE) for various baseline models has been established. These benchmarks are essential for objectively evaluating the performance of the advanced deep learning models that will be built in subsequent chapters.

### **Notebook 13: Dense and Recurrent Neural Networks (DNN & RNN)**

-   **Concept:** Moves beyond baselines to implement the first set of deep learning models. This notebook introduces two fundamental architectures: the simple **Linear Model** and the more complex **Deep Neural Network (DNN)**, also known as a Multi-Layer Perceptron (MLP). It applies these models to single-step, multi-step, and multi-output forecasting tasks, providing a comprehensive comparison of their capabilities.
-   **Dataset:** The preprocessed Metro Interstate Traffic Volume and Beijing Air Quality datasets.
-   **Analysis & Key Steps:**
    1.  **Reusing the `DataWindow` Class:** The powerful and reusable `DataWindow` class from the previous chapter is leveraged to prepare the data for all models, demonstrating its utility and saving significant coding effort. The notebook also includes parallel implementations in both **TensorFlow** and **PyTorch**.
    2.  **Linear Model:**
        -   **Architecture:** The simplest neural network, consisting of a single dense layer that learns a linear relationship between the inputs and outputs.
        -   **Tasks:** It is trained and evaluated on three distinct tasks:
            -   **Single-Step:** Predict the next hour of traffic volume.
            -   **Multi-Step:** Predict the next 24 hours of traffic volume.
            -   **Multi-Output:** Predict both traffic volume and temperature for the next hour.
        -   **Performance:** The linear model serves as a step up from the naive baselines, providing a more robust benchmark for the DNN.
    3.  **Deep Neural Network (DNN) Model:**
        -   **Architecture:** A more complex model featuring multiple hidden layers (in this case, two `Dense` layers with 64 neurons each) and the **ReLU activation function**. The non-linear activation allows the DNN to learn much more complex patterns than the linear model.
        -   **Tasks:** The DNN is applied to the same single-step, multi-step, and multi-output forecasting tasks as the linear model.
        -   **Performance:** In every scenario, the DNN significantly outperforms the linear model. This is because traffic volume has complex, non-linear relationships with time and weather, which the DNN's architecture is well-suited to capture.
    4.  **Comparative Evaluation:** The notebook concludes with a detailed performance comparison across all models (baselines, linear, and DNN) for each task. Bar charts are used to visualize the **Mean Absolute Error (MAE)** on both the validation and test sets.
        -   The results clearly show a hierarchy of performance: **DNN > Linear > Baselines**.
        -   This systematically demonstrates the value of adding complexity (hidden layers and non-linearities) to the model architecture when the underlying data patterns are non-linear.
    5.  **Exercises with Air Quality Data:** The concepts are reinforced through exercises where the same linear and DNN models are applied to the Beijing Air Quality dataset, tasking the user to predict `NO2` levels and showcasing the general applicability of these architectures.
