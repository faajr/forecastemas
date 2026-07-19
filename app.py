"""
Dashboard Streamlit untuk Forecasting Harga Emas.
Halaman:
  🏠 Dashboard  — ringkasan KPI & chart historis
  📈 Forecast   — pilih horizon & jalankan prediksi
  📊 Evaluasi   — perbandingan model (MAE / RMSE / MAPE)
  📄 Dataset    — tabel data historis
  ℹ️  About      — informasi proyek
"""

import io
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ─── Konstanta ────────────────────────────────────────────────────────────────
DATA_URL     = "https://raw.githubusercontent.com/faajr/forecastemas/main/harga_emas.csv"
DATA_PATH    = "harga_emas.csv"
MODEL_PATH   = "model.pkl"
METRICS_PATH = "metrics.json"

# Palet warna gold premium
GOLD_PRIMARY  = "#FFD700"
GOLD_DARK     = "#C9A000"
GOLD_LIGHT    = "#FFF0A0"
BG_DARK       = "#0D0D0D"
CARD_BG       = "rgba(255,215,0,0.06)"
CARD_BORDER   = "rgba(255,215,0,0.25)"
GREEN_UP      = "#00E676"
RED_DOWN      = "#FF5252"

# ─── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Forecast | Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Global (Dark Mode + Glassmorphism) ───────────────────────────────────
st.markdown("""
<style>
  /* ── Import font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D0D0D;
    color: #F0F0F0;
  }

  /* ── App background ── */
  .stApp {
    background: linear-gradient(135deg, #0D0D0D 0%, #1A1200 50%, #0D0D0D 100%);
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111 0%, #1C1400 100%) !important;
    border-right: 1px solid rgba(255,215,0,0.15);
  }
  [data-testid="stSidebar"] .stRadio label {
    color: #D4AF37 !important;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 6px 0;
    cursor: pointer;
    transition: color 0.2s;
  }
  [data-testid="stSidebar"] .stRadio label:hover {
    color: #FFD700 !important;
  }

  /* ── KPI Cards ── */
  .kpi-card {
    background: rgba(255,215,0,0.06);
    border: 1px solid rgba(255,215,0,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    backdrop-filter: blur(12px);
  }
  .kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(255,215,0,0.18);
  }
  .kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #A08800;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #FFD700;
    line-height: 1.1;
  }
  .kpi-sub {
    font-size: 0.75rem;
    color: #888;
    margin-top: 4px;
  }
  .kpi-up   { color: #00E676; }
  .kpi-down { color: #FF5252; }

  /* ── Section title ── */
  .section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFD700;
    letter-spacing: 0.04em;
    border-left: 4px solid #FFD700;
    padding-left: 10px;
    margin: 24px 0 14px;
  }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(120deg, rgba(255,215,0,0.12) 0%, rgba(201,160,0,0.06) 100%);
    border: 1px solid rgba(255,215,0,0.3);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    backdrop-filter: blur(16px);
  }
  .hero-icon {
    font-size: 3.5rem;
    line-height: 1;
  }
  .hero-title {
    font-size: 1.9rem;
    font-weight: 900;
    color: #FFD700;
    margin: 0 0 4px;
    letter-spacing: -0.02em;
  }
  .hero-sub {
    font-size: 0.92rem;
    color: #A08800;
    margin: 0;
  }

  /* ── Divider ── */
  .gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,215,0,0.4), transparent);
    margin: 24px 0;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #C9A000 0%, #FFD700 100%) !important;
    color: #0D0D0D !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
    width: 100%;
  }
  .stButton > button:hover {
    box-shadow: 0 6px 24px rgba(255,215,0,0.4) !important;
    transform: translateY(-2px) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }

  /* ── Radio buttons (horizon selector) ── */
  div[data-testid="stHorizontalBlock"] .stRadio > div {
    flex-direction: row;
    gap: 12px;
  }

  /* ── Streamlit default overrides ── */
  .stMetric {
    background: rgba(255,215,0,0.05);
    border-radius: 12px;
    padding: 12px !important;
    border: 1px solid rgba(255,215,0,0.15);
  }
  [data-testid="stMetricValue"] {
    color: #FFD700 !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricLabel"] {
    color: #888 !important;
  }

  /* ── DataFrame table ── */
  .stDataFrame { border-radius: 12px; overflow: hidden; }
  thead tr th {
    background: rgba(255,215,0,0.12) !important;
    color: #FFD700 !important;
  }

  /* ── Spinner ── */
  div.stSpinner > div {
    border-top-color: #FFD700 !important;
  }

  /* ── Info / Warning / Success boxes ── */
  .stAlert {
    border-radius: 12px !important;
    border-left-color: #FFD700 !important;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Muat & bersihkan dataset CSV dari URL atau file lokal."""
    if not os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_URL)
            df.to_csv(DATA_PATH, index=False)
        except Exception:
            return pd.DataFrame()
    else:
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            try:
                df = pd.read_csv(DATA_URL)
                df.to_csv(DATA_PATH, index=False)
            except Exception:
                return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.name in ["Date", "Price", "Tanggal"]:
        df = df.reset_index()

    col_date  = df.columns[0]
    col_price = df.columns[1]

    df[col_date]  = pd.to_datetime(df[col_date], errors="coerce")
    df            = df.dropna(subset=[col_date])
    df            = df.sort_values(col_date).reset_index(drop=True)
    df[col_price] = pd.to_numeric(df[col_price], errors="coerce")
    df[col_price] = df[col_price].ffill()
    df            = df.drop_duplicates(subset=[col_date])
    df            = df.set_index(col_date)
    df.index.name = "Date"
    df.columns    = ["Close"]
    return df


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}

def train_model_on_the_fly(df: pd.DataFrame):
    """Melatih model secara otomatis di server jika model.pkl tidak kompatibel atau tidak ada."""
    with st.spinner("⏳ Menyiapkan model machine learning di server (ini hanya dilakukan sekali)..."):
        TEST_SIZE = 30
        SEASONAL_PERIOD = 5
        
        train = df.iloc[:-TEST_SIZE]
        test  = df.iloc[-TEST_SIZE:]
        
        # 1. Holt-Winters
        try:
            model_hw = ExponentialSmoothing(
                train["Close"],
                trend="add",
                seasonal="add",
                seasonal_periods=SEASONAL_PERIOD,
                initialization_method="estimated",
            )
            fitted_hw = model_hw.fit(optimized=True)
            hw_preds  = fitted_hw.forecast(TEST_SIZE).values
            hw_metrics = compute_metrics(test["Close"].values, hw_preds)
        except Exception:
            hw_metrics = {"MAE": 999999.0, "RMSE": 999999.0, "MAPE": 100.0}
            fitted_hw = None
            
        # 2. Prophet
        try:
            df_prophet = train.reset_index().rename(columns={"Date": "ds", "Close": "y"})
            model_pr = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
            )
            model_pr.fit(df_prophet)
            future = model_pr.make_future_dataframe(periods=TEST_SIZE)
            forecast = model_pr.predict(future)
            pr_preds = forecast.tail(TEST_SIZE)["yhat"].values
            pr_metrics = compute_metrics(test["Close"].values, pr_preds)
        except Exception:
            pr_metrics = {"MAE": 999999.0, "RMSE": 999999.0, "MAPE": 100.0}
            model_pr = None
            
        # Tentukan terbaik
        if hw_metrics["MAPE"] <= pr_metrics["MAPE"] and fitted_hw is not None:
            best_model_name = "Holt-Winters"
            try:
                best_model = ExponentialSmoothing(
                    df["Close"],
                    trend="add",
                    seasonal="add",
                    seasonal_periods=SEASONAL_PERIOD,
                    initialization_method="estimated",
                ).fit(optimized=True)
            except Exception:
                best_model = fitted_hw
        else:
            best_model_name = "Prophet"
            try:
                full_prophet = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
                best_model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    changepoint_prior_scale=0.05,
                )
                best_model.fit(full_prophet)
            except Exception:
                best_model = model_pr
                
        # Simpan model
        model_data = {
            "model_name": best_model_name,
            "model":      best_model,
        }
        try:
            joblib.dump(model_data, MODEL_PATH)
        except Exception:
            pass
            
        # Simpan metrik
        last_price = float(df["Close"].iloc[-1])
        metrics_data = {
            "best_model":     best_model_name,
            "last_price":     round(last_price, 2),
            "last_date":      str(df.index[-1].date()),
            "holt_winters":   hw_metrics,
            "prophet":        pr_metrics,
        }
        try:
            with open(METRICS_PATH, "w") as f:
                json.dump(metrics_data, f, indent=2)
        except Exception:
            pass
            
        return model_data, metrics_data

def load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        if not df.empty:
            _, metrics_data = train_model_on_the_fly(df)
            return metrics_data
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        if not df.empty:
            _, metrics_data = train_model_on_the_fly(df)
            return metrics_data
        return {}

def load_model():
    if not os.path.exists(MODEL_PATH):
        if not df.empty:
            model_data, _ = train_model_on_the_fly(df)
            return model_data
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        if not df.empty:
            model_data, _ = train_model_on_the_fly(df)
            return model_data
        return None


def plotly_theme() -> dict:
    """Layout dasar untuk grafik Plotly dengan tema gelap gold."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#C0C0C0"),
        xaxis=dict(
            gridcolor="rgba(255,215,0,0.07)",
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,215,0,0.07)",
            zeroline=False,
            showline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,215,0,0.2)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="rgba(20,15,0,0.95)",
            bordercolor=GOLD_PRIMARY,
            font=dict(color="#FFD700", size=13),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def fmt_price(val: float, is_idr: bool, rate: float) -> str:
    """Format harga dinamis (USD / IDR)."""
    if pd.isna(val):
        return "—"
    converted = val * rate
    if is_idr:
        # Menghasilkan format Rp 16.000.000
        return f"Rp {int(converted):,}".replace(",", ".")
    return f"${converted:,.2f}"


def run_forecast(model_data: dict, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Menjalankan forecast dengan model yang sudah disimpan.
    Mengembalikan DataFrame dengan kolom: ds, yhat, yhat_lower, yhat_upper
    """
    model_name = model_data["model_name"]
    model      = model_data["model"]

    if model_name == "Prophet":
        future   = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        result   = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result   = result.reset_index(drop=True)

    else:  # Holt-Winters
        preds      = model.forecast(horizon)
        last_date  = df.index[-1]
        dates      = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        # Holt-Winters tidak punya interval bawaan, kira-kira ±2σ residu
        resid_std  = float(model.resid.std()) if hasattr(model, "resid") else float(preds.std() * 0.05)
        result     = pd.DataFrame({
            "ds":          dates,
            "yhat":        preds.values,
            "yhat_lower":  preds.values - 1.96 * resid_std,
            "yhat_upper":  preds.values + 1.96 * resid_std,
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar & Currency Settings
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
      <div style="font-size:2.8rem;">🥇</div>
      <div style="font-size:1.1rem; font-weight:800; color:#FFD700; letter-spacing:-0.02em;">
        Gold Forecast
      </div>
      <div style="font-size:0.75rem; color:#666; margin-top:4px;">ML · Time Series · Dashboard</div>
    </div>
    <hr style="border-color:rgba(255,215,0,0.15); margin:0 0 18px;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        options=[
            "🏠  Dashboard",
            "📈  Forecast",
            "📊  Evaluasi Model",
            "📄  Dataset",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )

    # -- Currency Toggle --
    st.markdown('<hr style="border-color:rgba(255,215,0,0.15); margin:18px 0 14px;">', unsafe_allow_html=True)
    st.markdown('<div style="color:#FFD700; font-weight:600; font-size:0.85rem; margin-bottom:8px;">MATA UANG</div>', unsafe_allow_html=True)
    currency_mode = st.radio("Mata Uang", ["USD", "IDR"], horizontal=True, label_visibility="collapsed")
    
    if currency_mode == "IDR":
        kurs_val = st.number_input("Kurs 1 USD = Rp", value=16200.0, step=100.0)
    else:
        kurs_val = 1.0
        
    is_idr = (currency_mode == "IDR")
    curr_label = "IDR" if is_idr else "USD"

    st.markdown("""
    <hr style="border-color:rgba(255,215,0,0.15); margin:18px 0 14px;">
    <div style="font-size:0.72rem; color:#555; text-align:center;">
      Data: GC=F via Yahoo Finance<br>
      Model: Holt-Winters · Prophet
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Data & model loading
# ═══════════════════════════════════════════════════════════════════════════════

df = load_data()
data_ok = not df.empty

if data_ok:
    metrics = load_metrics()
    try:
        model_data = load_model()
        model_ok = (model_data is not None)
    except Exception:
        model_ok = False
else:
    metrics = {}
    model_ok = False

if not data_ok:
    st.error(
        "⚠️ File **harga_emas.csv** tidak ditemukan di direktori ini dan gagal diunduh dari URL.\n\n"
        "Silakan periksa koneksi internet Anda lalu refresh halaman."
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if page.startswith("🏠"):
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-icon">🥇</div>
      <div>
        <p class="hero-title">Gold Price Forecast</p>
        <p class="hero-sub">Machine Learning · Holt-Winters &amp; Prophet ·</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.stop()

    last_price  = float(df["Close"].iloc[-1])
    prev_price  = float(df["Close"].iloc[-2])
    delta       = last_price - prev_price
    delta_pct   = delta / prev_price * 100
    delta_color = "kpi-up" if delta >= 0 else "kpi-down"
    delta_arrow = "▲" if delta >= 0 else "▼"

    best_model = metrics.get("best_model", "—")
    mape_val   = metrics.get("prophet" if best_model == "Prophet" else "holt_winters", {}).get("MAPE", "—")
    rmse_val   = metrics.get("prophet" if best_model == "Prophet" else "holt_winters", {}).get("RMSE", "—")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Harga Terakhir</div>
          <div class="kpi-value">{fmt_price(last_price, is_idr, kurs_val)}</div>
          <div class="kpi-sub {delta_color}">{delta_arrow} {fmt_price(abs(delta), is_idr, kurs_val)} ({delta_pct:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Model Terbaik</div>
          <div class="kpi-value" style="font-size:1.4rem;">{"—" if not model_ok else best_model}</div>
          <div class="kpi-sub">{"Latih model terlebih dahulu" if not model_ok else "Berdasarkan MAPE terendah"}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">MAPE</div>
          <div class="kpi-value">{f"{mape_val:.2f}%" if isinstance(mape_val, float) else "—"}</div>
          <div class="kpi-sub">Mean Abs Percentage Error</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        rmse_display = fmt_price(rmse_val, is_idr, kurs_val) if isinstance(rmse_val, (int, float)) else "—"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">RMSE</div>
          <div class="kpi-value" style="font-size:1.5rem;">{rmse_display}</div>
          <div class="kpi-sub">Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Chart historis
    st.markdown('<div class="section-title">📈 Grafik Harga Emas Historis</div>', unsafe_allow_html=True)

    year_min = int(df.index.year.min())
    year_max = int(df.index.year.max())
    sel_years = st.slider("Pilih rentang tahun", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)
    
    df_filtered = df[(df.index.year >= sel_years[0]) & (df.index.year <= sel_years[1])].copy()
    df_filtered["Close"] = df_filtered["Close"] * kurs_val  # Apply conversion to plot
    ma30 = df_filtered["Close"].rolling(30).mean()

    htemp = "<b>%{x|%d %b %Y}</b><br>Rp %{y:,.0f}<extra></extra>" if is_idr else "<b>%{x|%d %b %Y}</b><br>$%{y:,.2f}<extra></extra>"
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["Close"],
        mode="lines",
        name="Harga Aktual",
        line=dict(color=GOLD_PRIMARY, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,215,0,0.05)",
        hovertemplate=htemp,
    ))
    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=ma30,
        mode="lines",
        name="MA 30 Hari",
        line=dict(color="#FF8C00", width=1.8, dash="dash"),
        hovertemplate="MA30:<br>" + ("Rp %{y:,.0f}" if is_idr else "$%{y:,.2f}") + "<extra></extra>",
    ))
    
    y_prefix = "Rp " if is_idr else "$"
    fig_hist.update_layout(
        **plotly_theme(),
        height=420,
        title=dict(text=f"Harga Emas (GC=F) — {curr_label} per Troy Ounce", font=dict(color=GOLD_LIGHT, size=15)),
        xaxis_rangeslider_visible=True,
        yaxis=dict(tickprefix=y_prefix)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistik ringkas
    st.markdown('<div class="section-title">📊 Statistik Historis</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    stats = df["Close"].agg(["min", "max", "mean", "std"])
    s1.metric("Min", fmt_price(stats["min"], is_idr, kurs_val))
    s2.metric("Max", fmt_price(stats["max"], is_idr, kurs_val))
    s3.metric("Rata-rata", fmt_price(stats["mean"], is_idr, kurs_val))
    s4.metric("Std Dev", fmt_price(stats["std"], is_idr, kurs_val))
    s5.metric("Total Data", f"{len(df):,} hari")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Forecast
# ═══════════════════════════════════════════════════════════════════════════════

elif page.startswith("📈"):
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-icon">📈</div>
      <div>
        <p class="hero-title">Forecasting Harga Emas</p>
        <p class="hero-sub">Pilih horizon prediksi dan jalankan model ML</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok or not model_ok:
        st.warning("⚠️ Data/Model belum siap. Latih model terlebih dahulu.")
        st.stop()

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    with col_ctrl1:
        horizon = st.radio("Pilih Horizon Forecast", options=[7, 14, 30], format_func=lambda x: f"📅 {x} Hari", horizontal=True)
    with col_ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔮 Jalankan Forecast")

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    if run_btn:
        with st.spinner("⏳ Sedang menghitung forecast…"):
            model_data   = load_model()
            forecast_df  = run_forecast(model_data, df, horizon)

        model_name   = model_data["model_name"]
        last_price   = float(df["Close"].iloc[-1])
        last_date    = df.index[-1]
        
        yhat_max     = float(forecast_df["yhat"].max())
        yhat_min     = float(forecast_df["yhat"].min())
        yhat_mean    = float(forecast_df["yhat"].mean())

        st.markdown('<div class="section-title">📊 Ringkasan Forecast</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="kpi-card"><div class="kpi-label">Harga Hari Ini</div>
              <div class="kpi-value">{fmt_price(last_price, is_idr, kurs_val)}</div>
              <div class="kpi-sub">{last_date.strftime('%d %b %Y')}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card"><div class="kpi-label">Forecast Tertinggi</div>
              <div class="kpi-value kpi-up">{fmt_price(yhat_max, is_idr, kurs_val)}</div>
              <div class="kpi-sub">Dalam {horizon} hari</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card"><div class="kpi-label">Forecast Terendah</div>
              <div class="kpi-value kpi-down">{fmt_price(yhat_min, is_idr, kurs_val)}</div>
              <div class="kpi-sub">Dalam {horizon} hari</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card"><div class="kpi-label">Rata-rata Forecast</div>
              <div class="kpi-value">{fmt_price(yhat_mean, is_idr, kurs_val)}</div>
              <div class="kpi-sub">Dalam {horizon} hari</div>
            </div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="kpi-card"><div class="kpi-label">Model Digunakan</div>
              <div class="kpi-value" style="font-size:1.25rem;">{model_name}</div>
              <div class="kpi-sub">Model terbaik</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Grafik Historis + Forecast</div>', unsafe_allow_html=True)

        hist_tail = df.tail(90).copy()
        hist_tail["Close"] = hist_tail["Close"] * kurs_val
        
        fc_plot = forecast_df.copy()
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            fc_plot[col] = fc_plot[col] * kurs_val

        htemp_hist = "<b>%{x|%d %b %Y}</b><br>Rp %{y:,.0f}<extra></extra>" if is_idr else "<b>%{x|%d %b %Y}</b><br>$%{y:,.2f}<extra></extra>"
        htemp_fc   = "<b>%{x|%d %b %Y}</b><br>Forecast: " + ("Rp %{y:,.0f}" if is_idr else "$%{y:,.2f}") + "<extra></extra>"

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_plot["ds"], fc_plot["ds"][::-1]]),
            y=pd.concat([fc_plot["yhat_upper"], fc_plot["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,215,0,0.10)", line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Interval", hoverinfo="skip",
        ))
        fig_fc.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail["Close"], mode="lines",
            name="Historis", line=dict(color=GOLD_PRIMARY, width=1.8), hovertemplate=htemp_hist,
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_plot["ds"], y=fc_plot["yhat"], mode="lines+markers",
            name=f"Forecast {model_name}", line=dict(color="#FF8C00", width=2.5, dash="dot"),
            marker=dict(size=6, color="#FF8C00", symbol="circle"), hovertemplate=htemp_fc,
        ))

        fig_fc.add_vline(x=last_date, line_width=1.5, line_dash="dash", line_color="rgba(255,215,0,0.5)")
        y_prefix = "Rp " if is_idr else "$"
        fig_fc.update_layout(
            **plotly_theme(), height=460,
            title=dict(text=f"Forecast {horizon} Hari ke Depan — {model_name}", font=dict(color=GOLD_LIGHT, size=15)),
            xaxis_rangeslider_visible=False,
            yaxis=dict(tickprefix=y_prefix)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown('<div class="section-title">📋 Detail Forecast Harian</div>', unsafe_allow_html=True)
        tbl = forecast_df.copy()
        tbl["ds"]          = tbl["ds"].dt.strftime("%d %b %Y")
        tbl["yhat"]        = tbl["yhat"].apply(lambda x: fmt_price(x, is_idr, kurs_val))
        tbl["yhat_lower"]  = tbl["yhat_lower"].apply(lambda x: fmt_price(x, is_idr, kurs_val))
        tbl["yhat_upper"]  = tbl["yhat_upper"].apply(lambda x: fmt_price(x, is_idr, kurs_val))
        tbl.columns        = ["Tanggal", "Forecast", "Batas Bawah", "Batas Atas"]
        tbl.index          = range(1, len(tbl) + 1)
        st.dataframe(tbl, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Evaluasi Model
# ═══════════════════════════════════════════════════════════════════════════════

elif page.startswith("📊"):
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-icon">📊</div>
      <div>
        <p class="hero-title">Evaluasi Model</p>
        <p class="hero-sub">Perbandingan metrik Holt-Winters vs Prophet</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:
        st.warning("⚠️ File `metrics.json` tidak ditemukan. Latih model terlebih dahulu.")
        st.stop()

    hw  = metrics.get("holt_winters", {})
    pr  = metrics.get("prophet",      {})
    best = metrics.get("best_model",  "—")

    st.markdown(f"""
    <div style="background:rgba(255,215,0,0.08); border:1px solid rgba(255,215,0,0.3);
                border-radius:14px; padding:18px 24px; margin-bottom:24px;
                display:flex; align-items:center; gap:14px;">
      <span style="font-size:2rem;">🏆</span>
      <div>
        <div style="font-size:1rem; color:#888;">Model Terbaik (MAPE Terendah)</div>
        <div style="font-size:1.6rem; font-weight:800; color:#FFD700;">{best}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Tabel Perbandingan Metrik</div>', unsafe_allow_html=True)

    def fmt_metric(val, key, hw_v, pr_v, is_err_val=False):
        if not hw_v or not pr_v: return "—"
        hw_better = float(hw_v) <= float(pr_v)
        mark = "🟢" if (key == "hw" and hw_better) or (key == "pr" and not hw_better) else "🔴"
        
        # Jika is_err_val True (untuk MAE/RMSE), kalikan dengan kurs
        disp_val = float(val) * kurs_val if is_err_val else float(val)
        
        if is_err_val and is_idr:
            return f"{mark}  Rp {int(disp_val):,}".replace(",", ".")
        return f"{mark}  {disp_val:.4f}"

    rows = {
        "Metrik": [f"MAE ({curr_label})", f"RMSE ({curr_label})", "MAPE (%)"],
        "Holt-Winters": [
            fmt_metric(hw.get("MAE", 0),  "hw", hw.get("MAE"),  pr.get("MAE"), True),
            fmt_metric(hw.get("RMSE", 0), "hw", hw.get("RMSE"), pr.get("RMSE"), True),
            fmt_metric(hw.get("MAPE", 0), "hw", hw.get("MAPE"), pr.get("MAPE"), False),
        ],
        "Prophet": [
            fmt_metric(pr.get("MAE", 0),  "pr", hw.get("MAE"),  pr.get("MAE"), True),
            fmt_metric(pr.get("RMSE", 0), "pr", hw.get("RMSE"), pr.get("RMSE"), True),
            fmt_metric(pr.get("MAPE", 0), "pr", hw.get("MAPE"), pr.get("MAPE"), False),
        ],
    }
    st.dataframe(pd.DataFrame(rows).set_index("Metrik"), use_container_width=True)

    st.markdown('<div class="section-title">📈 Visualisasi Perbandingan MAPE</div>', unsafe_allow_html=True)
    
    # Hanya memvisualisasikan MAPE agar tidak terdistorsi satuan uang
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Bar(
        name="Holt-Winters", x=["MAPE (%)"], y=[hw.get("MAPE", 0)],
        marker_color=GOLD_PRIMARY, text=[f"{hw.get('MAPE', 0):.2f}%"], textposition="outside",
    ))
    fig_eval.add_trace(go.Bar(
        name="Prophet", x=["MAPE (%)"], y=[pr.get("MAPE", 0)],
        marker_color="#FF8C00", text=[f"{pr.get('MAPE', 0):.2f}%"], textposition="outside",
    ))
    fig_eval.update_layout(**plotly_theme(), barmode="group", height=380)
    st.plotly_chart(fig_eval, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — Dataset
# ═══════════════════════════════════════════════════════════════════════════════

elif page.startswith("📄"):
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-icon">📄</div>
      <div>
        <p class="hero-title">Dataset Harga Emas</p>
        <p class="hero-sub">GC=F — Harga penutupan harian</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.stop()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Data", f"{len(df):,} hari")
    col_b.metric("Mulai", df.index.min().strftime("%d %b %Y"))
    col_c.metric("Akhir",  df.index.max().strftime("%d %b %Y"))

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Filter & Tampilkan Data</div>', unsafe_allow_html=True)
    
    f1, f2 = st.columns(2)
    with f1:
        date_start = st.date_input("Dari tanggal", value=df.index.min().date())
    with f2:
        date_end   = st.date_input("Sampai tanggal", value=df.index.max().date())

    df_view = df[(df.index.date >= date_start) & (df.index.date <= date_end)].copy()
    df_view.index = df_view.index.strftime("%d %b %Y")
    
    col_name = f"Harga Penutupan ({curr_label})"
    df_view.columns = [col_name]
    df_view[col_name] = df_view[col_name].apply(lambda x: fmt_price(x, is_idr, kurs_val))

    st.dataframe(df_view, use_container_width=True, height=420)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — About 
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("ℹ️"):
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-icon">ℹ️</div>
      <div>
        <p class="hero-title">Tentang Proyek</p>
        <p class="hero-sub">Gold Price Forecast — Machine Learning Time Series</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 📌 Project Overview
    Proyek ini membangun sistem **forecasting harga emas** (GC=F — Gold Futures) menggunakan
    dua algoritma machine learning time series:

    - **Holt-Winters** (Triple Exponential Smoothing) — menangkap tren & musiman
    - **Prophet** (Meta/Facebook) — robust terhadap outlier & data harian

    Model terbaik dipilih otomatis berdasarkan **MAPE** terendah pada data uji.

    ---

    ## 📂 Dataset
    | Sumber | Yahoo Finance (`GC=F`) |
    |--------|----------------------|
    | Frekuensi | Harian (hari kerja) |
    | Rentang | Jul 2021 – sekarang |
    | Kolom | `Date`, `Close` (USD/oz) |

    ---

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

    ---

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

    ---

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

    ---

      Made with ❤️ ☕ · Gold Price Forecast Dashboard
      """, unsafe_allow_html=True)
