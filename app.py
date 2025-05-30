import os
import json
import paho.mqtt.client as mqtt
import mysql.connector
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS # <<< PENTING: Impor Flask-CORS
from dotenv import load_dotenv
from datetime import datetime, timedelta # Tambahkan timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
import threading # Sudah ada, bagus!

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # <<< PENTING: Mengaktifkan CORS untuk seluruh aplikasi Flask

# --- Database Configuration ---
# Pastikan variabel lingkungan ini sudah disetel di file .env Anda
DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "") # Sesuaikan dengan password Anda, pastikan ini ada di .env
DB_NAME = os.getenv("MYSQL_DB", "pool_monitor_db")

def get_db_connection():
    """Membuka koneksi ke database MySQL."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

# --- MQTT Configuration ---
MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC_SUBSCRIBE = os.getenv("MQTT_TOPIC_SUBSCRIBE", "pool/data")

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    """Callback saat klien MQTT terhubung ke broker."""
    if rc == 0:
        print(f"Connected to MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC_SUBSCRIBE)
        print(f"Subscribed to topic: {MQTT_TOPIC_SUBSCRIBE}")
    else:
        print(f"Failed to connect to MQTT Broker, return code {rc}")

def on_message(client, userdata, msg):
    """Callback saat pesan MQTT diterima."""
    try:
        payload = msg.payload.decode('utf-8') # Pastikan decoding UTF-8
        data = json.loads(payload)

        # Validasi bahwa semua kunci yang diperlukan ada
        ph = data.get("ph")
        turbidity = data.get("turbidity")
        temperature = data.get("temperature")

        if ph is None or turbidity is None or temperature is None:
            print(f"Warning: Missing data fields in payload: {payload}")
            return # Lewati jika data tidak lengkap

        print(f"Received data: pH={ph}, Turbidity={turbidity}, Temperature={temperature}")

        # Store data in MySQL
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            query = "INSERT INTO sensor_data (ph, turbidity, temperature) VALUES (%s, %s, %s)"
            cursor.execute(query, (ph, turbidity, temperature))
            conn.commit()
            cursor.close()
            conn.close()
            print("Data stored in MySQL successfully.")
        else:
            print("Could not store data: Database connection failed.")

    except json.JSONDecodeError:
        print(f"Error decoding JSON payload: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"Error processing MQTT message or storing to DB: {e}")

# --- Initialize MQTT Client ---
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1) # Menggunakan APIv1 untuk kompatibilitas
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start() # Start background thread for MQTT
    print("MQTT client started.")
except Exception as e:
    print(f"Failed to connect MQTT broker: {e}")

# --- Predictive Maintenance Function ---
def get_predictive_maintenance_recommendations():
    """
    Mengambil data sensor historis dan memberikan rekomendasi
    pemeliharaan prediktif berdasarkan tren.
    """
    conn = get_db_connection()
    if not conn:
        return {"status": "Database connection error for predictions."}

    cursor = conn.cursor(dictionary=True)
    # Ambil 100 data terbaru untuk prediksi
    cursor.execute("SELECT ph, turbidity, temperature, timestamp FROM sensor_data ORDER BY timestamp DESC LIMIT 100")
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    if not data or len(data) < 10: # Lebih banyak data lebih baik untuk regresi (misal: min 10)
        return {"status": "Not enough data for accurate predictive maintenance recommendations (min 10 readings required)."}

    # Urutkan data berdasarkan waktu dari yang terlama ke terbaru untuk regresi
    df = pd.DataFrame(data).sort_values(by='timestamp')
    df['timestamp_numeric'] = df['timestamp'].apply(lambda x: x.timestamp())

    recommendations = {}
    last_timestamp_numeric = df['timestamp_numeric'].iloc[-1]
    # Prediksi 1 jam ke depan
    predict_time_numeric = last_timestamp_numeric + 3600

    # pH Prediction
    if len(df['ph'].dropna()) >= 5:
        model_ph = LinearRegression()
        # Menggunakan semua data yang tersedia untuk melatih model
        X_ph = df['timestamp_numeric'].values.reshape(-1, 1)
        y_ph = df['ph'].values
        model_ph.fit(X_ph, y_ph)

        current_ph = df['ph'].iloc[-1] # pH saat ini (terbaru)
        predicted_ph_next_hour = model_ph.predict([[predict_time_numeric]])[0]

        if predicted_ph_next_hour < 7.2 and current_ph >= 7.2:
            recommendations['pH'] = "pH diprediksi akan turun di bawah level optimal segera. Pertimbangkan penambahan peningkat pH."
        elif predicted_ph_next_hour > 7.8 and current_ph <= 7.8:
            recommendations['pH'] = "pH diprediksi akan naik di atas level optimal segera. Pertimbangkan penambahan penurun pH."
        else:
            recommendations['pH'] = "pH stabil dan diprediksi tetap optimal."
    else:
        recommendations['pH'] = "Data pH tidak cukup untuk prediksi."

    # Turbidity Prediction
    if len(df['turbidity'].dropna()) >= 5:
        model_turbidity = LinearRegression()
        X_turb = df['timestamp_numeric'].values.reshape(-1, 1)
        y_turb = df['turbidity'].values
        model_turbidity.fit(X_turb, y_turb)

        current_turbidity = df['turbidity'].iloc[-1] # Turbidity saat ini
        predicted_turbidity_next_hour = model_turbidity.predict([[predict_time_numeric]])[0]

        if predicted_turbidity_next_hour > 5.0 and current_turbidity <= 5.0:
            recommendations['turbidity'] = "Kekeruhan diprediksi akan meningkat. Pertimbangkan backwash filter atau penambahan clarifier."
        else:
            recommendations['turbidity'] = "Kekeruhan stabil dan diprediksi tetap jernih."
    else:
        recommendations['turbidity'] = "Data kekeruhan tidak cukup untuk prediksi."

    # Temperature (biasanya untuk pemantauan, bukan prediksi tindakan proaktif kecuali ada sistem pemanas/pendingin)
    current_temperature = df['temperature'].iloc[-1]
    recommendations['temperature'] = f"Suhu air saat ini {current_temperature}°C. Pemantauan suhu aktif."


    if not recommendations:
        recommendations['status'] = "Parameter kualitas air diprediksi tetap optimal."
    elif len(recommendations) < 3: # Jika hanya beberapa rekomendasi, tambahkan status umum
        recommendations['status'] = "Beberapa rekomendasi pemeliharaan telah dibuat. Periksa detailnya."
        
    return recommendations


# --- Flask Routes ---

@app.route('/')
def index():
    """Menyajikan halaman HTML dashboard."""
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_sensor_data():
    """Mengembalikan data sensor terbaru dari database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Failed to connect to database"}), 500

    cursor = conn.cursor(dictionary=True)
    # Ambil 50 pembacaan terbaru
    cursor.execute("SELECT ph, turbidity, temperature, timestamp FROM sensor_data ORDER BY timestamp DESC LIMIT 50")
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Format timestamp untuk tampilan yang lebih baik di frontend
    # Catatan: `timestamp` dari MySQL bisa berupa objek datetime,
    # jadi `.strftime` akan berfungsi.
    formatted_data = []
    for row in data:
        if row['timestamp']:
            row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        formatted_data.append(row)

    return jsonify(formatted_data)

@app.route('/api/predictive_maintenance', methods=['GET'])
def get_predictions():
    """Mengembalikan rekomendasi pemeliharaan prediktif."""
    recommendations = get_predictive_maintenance_recommendations()
    return jsonify(recommendations)

if __name__ == '__main__':
    # Pastikan file index.html ada di dalam folder 'templates' di direktori yang sama
    # dengan app.py.
    # host='0.0.0.0' agar bisa diakses dari perangkat lain di jaringan lokal
    # debug=True untuk pengembangan (akan otomatis reload saat ada perubahan kode)
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)