"""
download_data.py
================
Script helper untuk mendownload data harga emas (GC=F)
dari Yahoo Finance dan menyimpannya sebagai CSV.

Jalankan:
  python3 download_data.py
"""

import pandas as pd
import yfinance as yf

OUTPUT = "harga_emas_siap_forecast.csv"

print("⬇️  Mendownload data harga emas (GC=F) dari Yahoo Finance...")
df = yf.download("GC=F", start="2021-07-19", progress=True)

# Ambil hanya kolom Close
df = df[["Close"]].copy()
df.index.name = "Price"
df.columns = ["Close"]

# Simpan
df.to_csv(OUTPUT)
print(f"✅ Data berhasil disimpan ke '{OUTPUT}' — {len(df)} baris")
print(f"   Rentang: {df.index.min()} – {df.index.max()}")
print(f"\nSekarang jalankan: python3 train_model.py")
