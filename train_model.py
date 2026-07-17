"""
train_model.py
==============
Pipeline pelatihan model untuk forecasting harga emas.

Langkah-langkah:
1. Membaca dataset CSV
2. Preprocessing data
3. Melatih model Holt-Winters
4. Melatih model Prophet
5. Evaluasi kedua model
6. Memilih model terbaik berdasarkan MAPE
7. Menyimpan model terbaik ke model.pkl
8. Menyimpan metrik evaluasi ke metrics.json
"""

import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

warnings.filterwarnings("ignore")

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
DATA_URL     = "https://raw.githubusercontent.com/faajr/forecastemas/main/harga_emas.csv"
DATA_PATH    = "harga_emas_siap_forecast.csv"
MODEL_PATH   = "model.pkl"
METRICS_PATH = "metrics.json"
TEST_SIZE    = 30          # Hari terakhir dipakai untuk evaluasi
SEASONAL_PERIOD = 5        # Periode musiman (hari kerja dalam seminggu)

# ─── 1. Membaca CSV ───────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Membaca dataset CSV dari URL atau file lokal dan mengembalikan DataFrame mentah."""
    print(f"[INFO] Mencoba mengunduh dataset dari URL: {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL)
        print(f"[INFO] Berhasil mengunduh dataset dari URL.")
        # Simpan ke file lokal untuk caching/Streamlit
        df.to_csv(path, index=False)
        print(f"[INFO] Dataset disimpan secara lokal di '{path}'.")
    except Exception as e:
        print(f"[WARNING] Gagal mengunduh dari URL: {e}")
        print(f"[INFO] Membaca dari file lokal: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File '{path}' tidak ditemukan secara lokal dan gagal mengunduh dari URL."
            )
        df = pd.read_csv(path)
    
    print(f"[INFO] Dataset berhasil dimuat: {len(df)} baris, kolom: {list(df.columns)}")
    return df



# ─── 2. Preprocessing ─────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dan memformat DataFrame:
    - Deteksi kolom tanggal & harga
    - Parsing tanggal
    - Drop baris non-tanggal
    - Forward-fill missing values
    - Sort ascending
    """
    df = df.copy()

    # Flatten MultiIndex jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Jika Date masih menjadi index, reset
    if df.index.name in ["Date", "Price", "Tanggal"]:
        df = df.reset_index()

    # Ambil kolom ke-0 sebagai tanggal, kolom ke-1 sebagai harga
    col_date  = df.columns[0]
    col_price = df.columns[1]

    # Parsing tanggal, paksa error → NaT
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")

    # Hapus baris yang bukan tanggal valid (misal header kedua)
    df = df.dropna(subset=[col_date])

    # Urutkan
    df = df.sort_values(col_date).reset_index(drop=True)

    # Konversi harga ke numerik
    df[col_price] = pd.to_numeric(df[col_price], errors="coerce")

    # Forward-fill missing value harga
    df[col_price] = df[col_price].ffill()

    # Drop duplikat
    df = df.drop_duplicates(subset=[col_date])

    # Set tanggal sebagai index
    df = df.set_index(col_date)
    df.index.name = "Date"
    df.columns    = ["Close"]

    print(f"[INFO] Setelah preprocessing: {len(df)} baris. "
          f"Rentang: {df.index.min().date()} – {df.index.max().date()}")
    return df


# ─── 3. Split train / test ────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: int):
    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]
    return train, test


# ─── 4a. Melatih Holt-Winters ─────────────────────────────────────────────────

def train_holtwinters(train: pd.DataFrame):
    """Melatih model Holt-Winters (Triple Exponential Smoothing)."""
    model = ExponentialSmoothing(
        train["Close"],
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    print("[INFO] Holt-Winters berhasil dilatih.")
    return fitted


def predict_holtwinters(fitted_model, steps: int) -> pd.Series:
    return fitted_model.forecast(steps)


# ─── 4b. Melatih Prophet ──────────────────────────────────────────────────────

def train_prophet(train: pd.DataFrame):
    """Melatih model Prophet."""
    df_prophet = train.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(df_prophet)
    print("[INFO] Prophet berhasil dilatih.")
    return model


def predict_prophet(model, last_date: pd.Timestamp, steps: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast.tail(steps)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


# ─── 5. Evaluasi ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


# ─── 6. Pilih model terbaik ───────────────────────────────────────────────────

def select_best(hw_metrics: dict, prophet_metrics: dict) -> str:
    """Memilih model dengan MAPE terkecil."""
    if hw_metrics["MAPE"] <= prophet_metrics["MAPE"]:
        return "Holt-Winters"
    return "Prophet"


# ─── 7 & 8. Simpan artefak ────────────────────────────────────────────────────

def save_artifacts(best_model_name: str, best_model, hw_metrics: dict,
                   prophet_metrics: dict, df: pd.DataFrame):
    # Simpan model
    joblib.dump(
        {
            "model_name": best_model_name,
            "model":      best_model,
        },
        MODEL_PATH,
    )
    print(f"[INFO] Model terbaik ({best_model_name}) disimpan ke '{MODEL_PATH}'.")

    # Simpan metrik
    last_price = float(df["Close"].iloc[-1])
    metrics = {
        "best_model":     best_model_name,
        "last_price":     round(last_price, 2),
        "last_date":      str(df.index[-1].date()),
        "holt_winters":   hw_metrics,
        "prophet":        prophet_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrik disimpan ke '{METRICS_PATH}'.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FORECAST HARGA EMAS — PIPELINE PELATIHAN MODEL")
    print("=" * 60)

    # 1. Load
    raw = load_data(DATA_PATH)

    # 2. Preprocessing
    df = preprocess(raw)

    # 3. Split
    train, test = split_data(df, TEST_SIZE)
    print(f"[INFO] Train: {len(train)} hari | Test: {len(test)} hari")

    # 4a. Holt-Winters
    hw_fitted = train_holtwinters(train)
    hw_preds  = predict_holtwinters(hw_fitted, TEST_SIZE).values
    hw_metrics = compute_metrics(test["Close"].values, hw_preds)
    print(f"[EVAL] Holt-Winters  → MAE={hw_metrics['MAE']:.2f}  "
          f"RMSE={hw_metrics['RMSE']:.2f}  MAPE={hw_metrics['MAPE']:.2f}%")

    # 4b. Prophet
    prophet_model  = train_prophet(train)
    prophet_result = predict_prophet(prophet_model, train.index[-1], TEST_SIZE)
    prophet_preds  = prophet_result["yhat"].values
    prophet_metrics = compute_metrics(test["Close"].values, prophet_preds)
    print(f"[EVAL] Prophet       → MAE={prophet_metrics['MAE']:.2f}  "
          f"RMSE={prophet_metrics['RMSE']:.2f}  MAPE={prophet_metrics['MAPE']:.2f}%")

    # 5. Pilih terbaik
    best_name = select_best(hw_metrics, prophet_metrics)
    print(f"\n[RESULT] Model terbaik: {best_name} 🏆")

    # Siapkan objek yang disimpan
    if best_name == "Prophet":
        # Latih ulang Prophet dengan seluruh data agar forecast lebih akurat
        best_model = train_prophet(df)
    else:
        best_model = train_holtwinters(df)

    # 6. Simpan artefak
    save_artifacts(best_name, best_model, hw_metrics, prophet_metrics, df)

    print("\n✅ Pipeline selesai! Sekarang jalankan: streamlit run app.py")


if __name__ == "__main__":
    main()
