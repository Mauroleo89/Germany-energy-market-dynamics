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
- **Prices_code_v1.ipynb** → the **pipeline and statistical foundation**, responsible for data collection, cleaning, feature engineering, and exploratory data analysis.  
- **Model Analysis.ipynb** → the **forecasting and evaluation framework**, where multiple models are implemented and compared.  

This repository serves as both an **academic contribution** and a **practical toolkit**:  
- For researchers → providing a comparative study of classical and modern forecasting methods.  
- For policymakers and businesses → offering insights to improve risk management and strategic planning in volatile fuel markets.  
- For data scientists → showcasing an end-to-end workflow, from raw data extraction to advanced model evaluation.  
