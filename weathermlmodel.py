import io, base64, socket, sqlite3, time, logging, traceback
import pandas as pd
import numpy as np
from flask import Flask, render_template
from matplotlib.figure import Figure
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("weathermlmodel.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIG ---
LAT, LON = 40.5432, -74.3632
LOCATION_NAME = "Metuchen, NJ"
DB_NAME = "weathermlmodelaccuracy.sqlite"
LAST_TRAIN_TIME = time.time()
SNOW_RATIO = 10.0
MM_TO_INCHES = 0.03937
FEATURES = ['temp', 'hum', 'press', 'temp_lag', 'hum_lag', 'press_lag']

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute('CREATE TABLE IF NOT EXISTS logs (timestamp TEXT PRIMARY KEY, predicted_mm REAL, actual_mm REAL)')
    conn.commit(); conn.close()

init_db()

cache_session = requests_cache.CachedSession('weathermlmodelcache.sqlite', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
regressor_model, classifier_model = None, None

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except: return "127.0.0.1"

def c_to_f(c): return (c * 9/5) + 32

def train_system():
    global regressor_model, classifier_model, LAST_TRAIN_TIME
    logger.info("🚀 Training Models...")
    hist_url = "https://archive-api.open-meteo.com/v1/archive"
    h_params = {"latitude": LAT, "longitude": LON, "start_date": "2023-01-01", "end_date": "2025-12-31",
                "hourly": ["precipitation", "temperature_2m", "relative_humidity_2m", "surface_pressure"]}
    responses = openmeteo.weather_api(hist_url, params=h_params)
    h = responses[0].Hourly()
    df = pd.DataFrame({
        "temp": h.Variables(1).ValuesAsNumpy(),
        "hum": h.Variables(2).ValuesAsNumpy(),
        "press": h.Variables(3).ValuesAsNumpy(),
        "target_amt": h.Variables(0).ValuesAsNumpy()
    })
    df['target_is_rain'] = (df['target_amt'] > 0.1).astype(int)
    for col in ['temp', 'hum', 'press']: df[f'{col}_lag'] = df[col].shift(1)
    df = df.dropna()
    regressor_model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(df[FEATURES], df['target_amt'])
    classifier_model = LogisticRegression(max_iter=1000).fit(df[FEATURES], df['target_is_rain'])
    LAST_TRAIN_TIME = time.time()

def sync_actual_data():
    conn = sqlite3.connect(DB_NAME)
    hist_url = "https://archive-api.open-meteo.com/v1/archive"
    yesterday = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    params = {"latitude": LAT, "longitude": LON, "start_date": yesterday, "end_date": today, "hourly": "precipitation"}
    res = openmeteo.weather_api(hist_url, params=params)[0].Hourly()
    time_range = range(res.Time(), res.TimeEnd(), res.Interval())
    timestamps = [str(pd.to_datetime(t, unit="s")) for t in time_range]
    actuals = pd.DataFrame({"timestamp": timestamps, "actual_mm": res.Variables(0).ValuesAsNumpy()})
    for _, row in actuals.iterrows():
        conn.execute("UPDATE logs SET actual_mm = ? WHERE timestamp = ?", (float(row['actual_mm']), row['timestamp']))
    conn.commit(); conn.close()

def get_forecast_and_log():
    f_url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": LAT, "longitude": LON, "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure"], "forecast_days": 10}
    f_res = openmeteo.weather_api(f_url, params=params)[0].Hourly()
    time_range = range(f_res.Time(), f_res.TimeEnd(), f_res.Interval())
    df_f = pd.DataFrame({
        "time": [pd.to_datetime(t, unit="s") for t in time_range],
        "temp": f_res.Variables(0).ValuesAsNumpy(),
        "hum": f_res.Variables(1).ValuesAsNumpy(),
        "press": f_res.Variables(2).ValuesAsNumpy()
    })
    for col in ['temp', 'hum', 'press']: df_f[f'{col}_lag'] = df_f[col].shift(1).fillna(df_f[col].iloc[0])
    df_f['pred_amt_mm'] = regressor_model.predict(df_f[FEATURES])
    df_f['pred_prob'] = classifier_model.predict_proba(df_f[FEATURES])[:, 1] * 100
    df_f['temp_f'] = c_to_f(df_f['temp'])
    df_f['pred_inches'] = df_f['pred_amt_mm'] * MM_TO_INCHES
    df_f['is_snow'] = df_f['temp'] < 1.0
    df_f['snow_inches'] = np.where(df_f['is_snow'], df_f['pred_inches'] * SNOW_RATIO, 0)
    df_f['cumulative_snow'] = df_f['snow_inches'].cumsum()
    conn = sqlite3.connect(DB_NAME)
    for _, row in df_f.iterrows():
        conn.execute("INSERT OR REPLACE INTO logs (timestamp, predicted_mm, actual_mm) VALUES (?, ?, (SELECT actual_mm FROM logs WHERE timestamp=?))",
                     (str(row['time']), float(row['pred_amt_mm']), str(row['time'])))
    conn.commit(); conn.close()
    return df_f

@app.route('/')
def index():
    try:
        global LAST_TRAIN_TIME
        if (time.time() - LAST_TRAIN_TIME) > 604800: train_system()
        sync_actual_data(); df_f = get_forecast_and_log()
        snow_mode = df_f.iloc[0:48]['is_snow'].any()
        conn = sqlite3.connect(DB_NAME); perf = pd.read_sql("SELECT * FROM logs WHERE actual_mm IS NOT NULL", conn)
        mae_in = round(np.mean(np.abs(perf['predicted_mm'] - perf['actual_mm'])) * MM_TO_INCHES, 4) if not perf.empty else "N/A"
        yesterday_data = perf.tail(24)
        if len(yesterday_data) >= 12:
            diff_in = (yesterday_data['predicted_mm'].sum() - yesterday_data['actual_mm'].sum()) * MM_TO_INCHES
            summary = f"Off by {abs(round(diff_in,3))} inches"
        else: summary = "Calibrating..."
        conn.close()

        fig = Figure(figsize=(12, 6), facecolor='#16161a')
        ax1 = fig.subplots(); ax1.set_facecolor('#16161a'); ax2 = ax1.twinx()
        if snow_mode:
            ax1.plot(df_f['time'], df_f['snow_inches'], color='#ffffff', linewidth=2, label='Hourly Snow')
            ax1.plot(df_f['time'], df_f['cumulative_snow'], color='#00ff88', linestyle=':', label='Total Depth')
            ax1.set_ylabel("Snow Depth (Inches)", color="#ffffff")
        else:
            ax1.plot(df_f['time'], df_f['pred_inches'], color='#00d4ff', linewidth=3, label='Rain')
            ax1.set_ylabel("Rain (Inches)", color="#00d4ff")
        ax2.plot(df_f['time'], df_f['pred_prob'], color='#ff007a', linestyle='--', alpha=0.4, label='Rain Chance %')
        ax2.plot(df_f['time'], df_f['temp_f'], color='#ffcc00', linewidth=1.5, label='Temp (°F)')
        ax2.set_ylabel("Prob (%) / Temp (°F)", color="#ffcc00")
        ax1.tick_params(colors='white'); ax2.tick_params(axis='y', labelcolor='#ffcc00')
        ax1.grid(alpha=0.1); ax2.set_ylim(0, max(105, df_f['temp_f'].max() + 10))
        ax1.legend(loc='upper left', facecolor='#16161a', labelcolor='white')
        ax2.legend(loc='upper right', facecolor='#16161a', labelcolor='white')
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight')
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return render_template('index.html', plot_url=plot_url, mae_in=mae_in, location=LOCATION_NAME,
                               snow_mode=snow_mode, summary=summary, total_snow=df_f['snow_inches'].sum(),
                               min_t=round(df_f['temp_f'].min(),1), max_t=round(df_f['temp_f'].max(),1),
                               ip_addr=get_local_ip(), time=pd.Timestamp.utcnow().strftime('%H:%M:%S'))
    except: return f"<pre>{traceback.format_exc()}</pre>", 500

@app.route('/design')
def design():
    return render_template('design.html')

if __name__ == '__main__':
    train_system(); app.run(host='0.0.0.0', port=5000)
