
# 🚦 AI-Driven Traffic Flow Optimization

## 📌 Overview

This project focuses on **forecasting traffic speeds** on different road segments in Los Angeles using the **METR-LA dataset**. The goal is to leverage **time-series forecasting** and **machine learning** to provide actionable insights for **traffic management and urban planning**.

We built a **Streamlit dashboard** that allows users to:

* Select a **road sensor ID**
* Choose a **sequence length (past time window)**
* Generate **future traffic flow predictions**
* Visualize **actual vs predicted traffic speeds**
* 
## 📂 Dataset
We use the **[METR-LA dataset](https://github.com/liyaguang/DCRNN)**, which contains:

* Traffic speed data collected from **207 loop detectors** on highways in Los Angeles.
* Data recorded every **5 minutes** over a **4-month period (March–June 2012)**.
* Format: HDF5 (`.h5`) file.

## ⚙️ Project Structure
📁 traffic-flow-optimization
 ├── app.py                 # Streamlit dashboard  
 ├── AI_Traffic_Flow_Optimization.ipynb   # Model training & exploration notebook  
 ├── METR-LA.h5             # Traffic dataset (not included in repo)  
 ├── requirements.txt       # Python dependencies  
 ├── README.md              # Project documentation  
 
## 📊 Features

* **Interactive Dashboard**: Choose **Sensor ID** and **Sequence Length** dynamically.
* **Visualization**: Displays real traffic speeds vs. model predictions.
* **Scalable**: Can be extended to support multiple forecasting models (LSTM, GRU, DCRNN).
* 
## 🔮 Future Enhancements

* Incorporate **weather and event data** for better forecasting accuracy.
* Deploy the system on **AWS/GCP for real-time streaming predictions**.
* Implement **multi-step ahead forecasting**.
* Extend to **AI-driven traffic light optimization** using reinforcement learning.

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy, TensorFlow/PyTorch, h5py)
* **Streamlit** (Dashboard UI)
* **scikit-learn** (Preprocessing, Evaluation)
* **HDF5** (Dataset storage format)

## 👨‍💻 Author

* Mahak  (AI & Data Science Student)
