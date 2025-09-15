# A Machine Learning Approach for Analyzing Energy Market Dynamics

This repository contains the code, analysis, and documentation from my project **“A Machine Learning Approach for Analyzing Energy Market Dynamics”**. The project forecasts **diesel, E5, and E10 prices** in Germany using statistical, machine learning, and deep learning methods.  

---

## 📌 Project Description  

Fuel price volatility in Germany — particularly for **diesel, E5, and E10 gasoline** — has become a critical issue due to its impact on consumers, businesses, and national policy. Sudden changes in international oil markets, exchange rates, and electricity prices create uncertainty that affects transportation costs, supply chains, and energy transition strategies.  

This project addresses these challenges by:  
- Building a **data pipeline** to collect, clean, and process information from multiple sources (Tankerkönig API, EIA, Bundesnetzagentur, ECB).  
- Performing **exploratory statistical analysis** to understand fuel price behavior across German states and in response to international drivers.  
- Developing and comparing **forecasting models**:  
  - **Statistical models** → ARIMA, VAR  
  - **Machine Learning models** → Random Forest, XGBoost  
  - **Deep Learning models** → LSTM, GRU  
- Evaluating models with metrics such as **MSE, RMSE, MAE, and Accuracy**.  
- Identifying the most relevant factors influencing fuel prices, including international crude oil benchmarks (WTI, Brent), exchange rates (EUR/USD), electricity generation, and financial indices (DAX40, S&P500).  

The project is divided into two complementary components:  
- **Data_Analysis.ipynb** → the **pipeline and statistical foundation**, responsible for data collection, cleaning, feature engineering, and exploratory data analysis.  
- **Model Analysis.ipynb** → the **forecasting and evaluation framework**, where multiple models are implemented and compared.  

This repository serves as both an **academic contribution** and a **practical toolkit**:  
- For researchers → providing a comparative study of classical and modern forecasting methods.  
- For policymakers and businesses → offering insights to improve risk management and strategic planning in volatile fuel markets.  
- For data scientists → showcasing an end-to-end workflow, from raw data extraction to advanced model evaluation.  

---

## 🎯 Problem & Context  

Fuel prices in Germany are highly volatile, influenced by international crude oil markets, exchange rates, electricity prices, and geopolitical shocks.  
This volatility directly affects **consumers** (household costs), **businesses** (transportation and supply chains), and **policymakers** (energy security, inflation control, and transition strategies).  

### Challenges Addressed:
- Daily fluctuations in **diesel, E5, and E10 gasoline** prices with no clear patterns.  
- Limited predictive capacity of traditional econometric models in capturing sudden shocks.  
- Need for more **transparent and accurate forecasting tools** to support decision-making in the German energy market.  

---

## 🎯 Objectives & Research Questions  

**Main Goal:**  
To develop and evaluate predictive models for short-term fuel prices in Germany, using both domestic and international market indicators.  

**Specific Objectives:**  
1. Create a **data pipeline** to gather and preprocess fuel and energy data from diverse sources.  
2. Apply **statistical, machine learning, and deep learning models** to forecast daily fuel prices.  
3. Compare model performance across accuracy metrics (MSE, RMSE, MAE, Accuracy).  
4. Identify the **key drivers** of fuel price fluctuations in Germany.  
5. Provide insights for **strategic planning and risk management** in volatile fuel markets.  

**Research Question:**  
*Which supervised learning models are most suitable for predicting daily changes in German fuel prices (diesel, E5, E10), and how does their performance differ when incorporating additional variables such as international oil prices, exchange rates, and electricity generation?*  

---

## 🔄 Data Pipeline & Sources  

---

## 🔄 Data Pipeline & Model Analysis  

This project is structured into two complementary parts, following the methodology described in the thesis:

1. **Data Analysis & Pipeline (📓 `Data_Analysis.ipynb`)**  
2. **Modeling & Evaluation (📓 `Model Analysis.ipynb`)**

Together, these components form an end-to-end framework for forecasting German fuel prices.

---

### 1️⃣ Data Analysis & Pipeline (`Data_Analysis.ipynb`)

This notebook builds the **statistical foundation** of the project by creating a robust dataset from multiple heterogeneous sources. It is directly inspired by the **Research Design and Methods** section of the thesis:contentReference[oaicite:0]{index=0}.

#### Main Steps:
- **Data Collection**  
  - Fuel prices (diesel, E5, E10) from the **Tankerkönig API** (2014–2024).  
  - International oil and gas prices (WTI, Brent, Henry Hub) from **EIA**.  
  - Electricity generation & consumption (renewables vs conventional) from **Bundesnetzagentur**.  
  - Exchange rates (EUR/USD, GBP/USD) from **ECB**.  
  - Financial indices (DAX40, S&P500) from **Dukascopy**.

- **Processing**  
  - Cleaning and imputing missing values (holiday gaps filled with last available values).  
  - Aggregating hourly data into **daily averages** at state and national level.  
  - Normalizing electricity generation to GWh and splitting renewable vs conventional.  
  - Feature engineering: lagged variables, average electricity prices, currency fluctuations.

- **Exploratory Data Analysis (EDA)**  
  - **Descriptive statistics** (mean, min, max, std per fuel type).  
  - **Correlation heatmaps** linking German fuel prices with WTI, Brent, EUR/USD, and electricity prices.  
  - **Comparative analysis across states** (ANOVA test confirms both regional and seasonal influences).  
  - **Visualizations**: historical fuel price evolution, histograms, regional differences, COVID-19 and Ukraine war impacts.

📌 **Output:** A structured dataset with enriched features, ready for predictive modeling.  

---

### 2️⃣ Model Analysis (`Model Analysis.ipynb`)

This notebook implements and compares **statistical, machine learning, and deep learning models**, as described in the **Model Selection and Results** chapters of the thesis:contentReference[oaicite:1]{index=1}.

#### Workflow:
1. **Data Preparation**  
   - Focused on **Berlin diesel prices** as representative of national trends.  
   - Scaled features (MinMaxScaler).  
   - Created lag features (past 7–30 days).  
   - Train/test split (80% train, 20% test, preserving time order).

2. **Modeling Approaches**
   - **Statistical Models**  
     - *ARIMA*: parameter tuning with cross-validation; best fit ARIMA(5,1,0).  
     - *VAR*: multivariate system including diesel, E5, E10, oil benchmarks, exchange rates, and financial indices.  
   - **Machine Learning Models**  
     - *Random Forest*: hyperparameter tuning (tree depth, estimators).  
     - *XGBoost*: tuned learning rate, subsample, iterations.  
   - **Deep Learning Models**  
     - *LSTM*: two stacked layers, 30-day sequences, trained with Adam optimizer.  
     - *GRU*: simplified recurrent architecture, faster convergence, superior generalization.

3. **Evaluation Metrics**  
   - Mean Squared Error (MSE)  
   - Root Mean Squared Error (RMSE)  
   - Mean Absolute Error (MAE)  
   - Accuracy (±1 cent tolerance)  

4. **Model Comparison**  
   - **ARIMA**: strong for short-term forecasts, but weak on shocks.  
   - **VAR**: best at capturing multivariate dependencies.  
   - **Random Forest & XGBoost**: robust, interpretable, competitive accuracy.  
   - **LSTM**: underperformed due to complexity and overfitting.  
   - **GRU**: best deep learning performer, combining efficiency and accuracy.

📌 **Output:** Comparative tables, forecasts vs. actual plots, and discussion of each model’s strengths and limitations.

---

### 🔑 Integrated Findings

- International oil prices (**WTI & Brent**) and exchange rates (**EUR/USD**) are the **strongest predictors** of German fuel prices.  
- **Electricity prices** show moderate correlation, indicating substitution effects with electromobility.  
- **VAR** and **GRU** achieved the best results overall, with GRU excelling in capturing non-linear dependencies.  
- **Random Forest** and **XGBoost** are strong alternatives for interpretable, reliable forecasts.  
- **LSTM** was less effective in this dataset, confirming the importance of choosing architectures carefully.

---
