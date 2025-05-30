import os
import json
import paho.mqtt.client as mqtt
import mysql.connector
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Database Configuration ---
DB_HOST = os.getenv("MYSQL_HOST","localhost")
DB_USER = os.getenv("MYSQL_USER","root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD","G16th07b")
DB_NAME = os.getenv("MYSQL_DB","pool_monitor_db")

def get_db_connection():
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    return conn

# --- MQTT Configuration ---
MQTT_BROKER = os.getenv("MQTT_BROKER","broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT",1883))
MQTT_TOPIC_SUBSCRIBE = os.getenv("MQTT_TOPIC_SUBSCRIBE","pool/data")

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe(MQTT_TOPIC_SUBSCRIBE)
    print(f"Subscribed to topic: {MQTT_TOPIC_SUBSCRIBE}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        ph = data.get("ph")
        turbidity = data.get("turbidity")
        temperature = data.get("temperature")

        print(f"Received data: pH={ph}, Turbidity={turbidity}, Temperature={temperature}")

        # Store data in MySQL
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO sensor_data (ph, turbidity, temperature) VALUES (%s, %s, %s)"
        cursor.execute(query, (ph, turbidity, temperature))
        conn.commit()
        cursor.close()
        conn.close()
        print("Data stored in MySQL successfully.")

    except json.JSONDecodeError:
        print(f"Error decoding JSON payload: {msg.payload}")
    except Exception as e:
        print(f"Error processing MQTT message or storing to DB: {e}")

# --- Initialize MQTT Client ---
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start() # Start background thread for MQTT


# --- Predictive Maintenance Function ---
def get_predictive_maintenance_recommendations():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT ph, turbidity, temperature, timestamp FROM sensor_data ORDER BY timestamp DESC LIMIT 100") # Get last 100 readings
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    if not data or len(data) < 5: # Need enough data for prediction
        return {"status": "Not enough data for accurate prediction."}

    df = pd.DataFrame(data)
    df['timestamp_numeric'] = df['timestamp'].apply(lambda x: x.timestamp())

    recommendations = {}

    # pH Prediction
    if len(df['ph'].dropna()) >= 5: # Ensure enough non-null values
        model_ph = LinearRegression()
        X_ph = df['timestamp_numeric'].values.reshape(-1, 1)
        y_ph = df['ph'].values
        model_ph.fit(X_ph, y_ph)
        
        last_ph = df['ph'].iloc[0]
        predicted_ph_next_hour = model_ph.predict([[df['timestamp_numeric'].iloc[0] + 3600]]) # Predict 1 hour later
        
        if predicted_ph_next_hour < 7.2 and last_ph >= 7.2:
            recommendations['ph'] = "pH is predicted to drop below optimal levels soon. Consider adding pH Increaser."
        elif predicted_ph_next_hour > 7.8 and last_ph <= 7.8:
            recommendations['ph'] = "pH is predicted to rise above optimal levels soon. Consider adding pH Reducer."
        else:
            recommendations['ph'] = "pH is stable."
    else:
        recommendations['ph'] = "Insufficient data for pH prediction."


    # Turbidity Prediction
    if len(df['turbidity'].dropna()) >= 5:
        model_turbidity = LinearRegression()
        X_turb = df['timestamp_numeric'].values.reshape(-1, 1)
        y_turb = df['turbidity'].values
        model_turbidity.fit(X_turb, y_turb)

        last_turbidity = df['turbidity'].iloc[0]
        predicted_turbidity_next_hour = model_turbidity.predict([[df['timestamp_numeric'].iloc[0] + 3600]])

        if predicted_turbidity_next_hour > 5.0 and last_turbidity <= 5.0: # Example threshold for turbidity
            recommendations['turbidity'] = "Turbidity is predicted to increase. Consider backwashing filter or adding clarifier."
        else:
            recommendations['turbidity'] = "Turbidity is stable."
    else:
        recommendations['turbidity'] = "Insufficient data for turbidity prediction."

    # Temperature (less critical for predictive maintenance, more for monitoring)
    recommendations['temperature'] = "Temperature monitoring is active."

    return recommendations


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_sensor_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT ph, turbidity, temperature, timestamp FROM sensor_data ORDER BY timestamp DESC LIMIT 50") # Get last 50 readings
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Format timestamp for better display in frontend
    for row in data:
        if row['timestamp']:
            row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

    return jsonify(data)

@app.route('/api/predictive_maintenance', methods=['GET'])
def get_predictions():
    recommendations = get_predictive_maintenance_recommendations()
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)