from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import os
from io import BytesIO
import base64

app = Flask(__name__)

# === Muat model dan scaler ===
model = load_model("model_tesla.h5")
scaler = joblib.load("scaler_tesla.pkl")

# === Muat dataset Tesla untuk ditampilkan grafiknya ===
df = pd.read_csv("Tesla.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

@app.route('/')
def index():
    # Buat grafik historis harga penutupan
    plt.figure(figsize=(8,4))
    plt.plot(df['Date'], df['Close'], label='Harga Penutupan')
    plt.title('Harga Saham Tesla (Historis)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga (USD)')
    plt.legend()
    plt.tight_layout()

    # Simpan grafik ke base64 agar bisa ditampilkan di HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('index.html', plot_url=plot_url)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil jumlah hari terakhir dari form
        n_days = int(request.form['n_days'])
        if n_days <= 0:
            return render_template('index.html', error="Jumlah hari harus lebih dari 0")

        # Ambil data 'Close' terakhir dari dataset
        close_data = df[['Close']].values
        scaled_data = scaler.transform(close_data)

        # Gunakan data terakhir untuk prediksi
        last_sequence = scaled_data[-30:].reshape(1, 30, 1)  # 30 timesteps terakhir
        prediction_scaled = model.predict(last_sequence)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        # Tampilkan hasil
        result = f"Prediksi harga saham Tesla untuk hari ke-{n_days} berikutnya: ${prediction:.2f}"

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
