## ℹ️Gold Price Forecast — Machine Learning Time Series
## 📌 Project Overview
    Proyek ini membangun sistem **forecasting harga emas** (GC=F — Gold Futures) menggunakan
    dua algoritma machine learning time series:

    - **Holt-Winters** (Triple Exponential Smoothing) — menangkap tren & musiman
    - **Prophet** (Meta/Facebook) — robust terhadap outlier & data harian

    Model terbaik dipilih otomatis berdasarkan **MAPE** terendah pada data uji.

## 📂 Dataset
    | Sumber | Yahoo Finance (`GC=F`) |
    |--------|----------------------|
    | Frekuensi | Harian (hari kerja) |
    | Rentang | Jul 2021 – sekarang |
    | Kolom | `Date`, `Close` (USD/oz) |

## 🔬 Machine Learning Workflow

    ```
      harga_emas.csv
            │
            ▼
      train_model.py
            │
      ┌─────┴──────┐
      │            │
    Holt-Winters  Prophet
      │            │
      └─────┬──────┘
            │
       Evaluasi & Pilih Terbaik
            │
       model.pkl + metrics.json
            │
            ▼
         app.py (Dashboard)
    ```

## 🚀 Cara Menjalankan

    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Letakkan file CSV di direktori ini
    #    harga_emas_siap_forecast.csv

    # 3. Latih model
    python train_model.py

    # 4. Jalankan dashboard
    streamlit run app.py
    ```

 ## 🌐 Deployment
    Dapat di-deploy ke **Streamlit Community Cloud**:
    1. Push repo ke GitHub
    2. Login ke [share.streamlit.io](https://share.streamlit.io)
    3. Deploy dari repo

    ---
## 🛠️ Tech Stack
    | Library | Versi |
    |---------|-------|
    | streamlit | ≥ 1.30 |
    | plotly | ≥ 5.0 |
    | prophet | ≥ 1.1 |
    | statsmodels | ≥ 0.14 |
    | pandas | ≥ 1.5 |
    | numpy | ≥ 1.23 |
    | joblib | ≥ 1.2 |
    | scikit-learn | ≥ 1.2 |


Made with ❤️ ☕ · Gold Price Forecast Dashboard
