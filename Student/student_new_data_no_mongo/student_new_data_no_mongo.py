import os
import sys
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from flask import Flask, request, jsonify, send_from_directory

# --- Configuration ---
DATA_FILE = r"Student/data/StudentPerformanceFactors.csv"
ARTIFACTS_DIR = Path("./artifacts")
WEB_DIR = Path("./web")
MODEL_PATH = ARTIFACTS_DIR / "best_model.keras"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(exist_ok=True)
WEB_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(WEB_DIR))

# --- Data Loading & Processing ---
print("\n[INIT] Loading dataset...")
if not os.path.exists(DATA_FILE):
    print(f"ERROR: dataset not found at: {DATA_FILE}")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
df = df.drop_duplicates().ffill()

target = "Exam_Score"
if target not in df.columns:
    print("ERROR: Target column 'Exam_Score' not found in dataset")
    sys.exit(1)

X_df = df.drop(columns=[target])
y = df[target]

cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"[DATA] Numeric columns: {num_cols}")
print(f"[DATA] Categorical columns: {cat_cols}")

# --- Preprocessor Setup ---
from sklearn import __version__ as skl
_major = int(skl.split(".")[0])
_minor = int(skl.split(".")[1]) if len(skl.split(".")) > 1 else 0

if _major >= 1 and _minor >= 2:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", ohe, cat_cols)])

if PREPROCESSOR_PATH.exists():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("[INIT] Loaded preprocessor from cache")
else:
    preprocessor = preprocessor.fit(X_df)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("[INIT] Fitted and saved preprocessor")

X_processed = preprocessor.transform(X_df)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- Classical Models Training ---
model_performance = {}
print("[TRAIN] Training classical models...")

lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=18,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
).fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

model_performance['Linear Regression'] = {
    'MAE': mean_absolute_error(y_test, lr_pred),
    'MSE': mean_squared_error(y_test, lr_pred),
    'R2': r2_score(y_test, lr_pred)
}

model_performance['Random Forest'] = {
    'MAE': mean_absolute_error(y_test, rf_pred),
    'MSE': mean_squared_error(y_test, rf_pred),
    'R2': r2_score(y_test, rf_pred)
}

# --- Neural Network Setup ---
input_dim = X_train.shape[1]

def build_nn(input_dim):
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0015)
    m.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return m

print("\n[TRAIN] Setting up Neural Network...")
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INIT] Loaded NN model from: {MODEL_PATH}")
else:
    model = build_nn(input_dim)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    cp = callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), save_best_only=True)

    print("[TRAIN] Starting Neural Network training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=250,
        batch_size=64,
        callbacks=[es, rlr, cp],
        verbose=1
    )
    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history.history, f)
    print(f"[TRAIN] Model saved to: {MODEL_PATH}")

nn_preds = model.predict(X_test).flatten()
model_performance['Neural Network'] = {
    'MAE': mean_absolute_error(y_test, nn_preds),
    'MSE': mean_squared_error(y_test, nn_preds),
    'R2': r2_score(y_test, nn_preds)
}

# --- PRINTING INITIAL METRICS TO TERMINAL ---
print("\n" + "█"*50)
print("📊 FINAL MODEL PERFORMANCE REPORT (ON SERVER START)")
print("█"*50)
for name, metrics in model_performance.items():
    print(f"🔹 {name}:")
    print(f"   ├─ R² Score: {metrics['R2']:.4f}")
    print(f"   ├─ MAE:      {metrics['MAE']:.4f}")
    print(f"   └─ MSE:      {metrics['MSE']:.4f}")
print("█"*50 + "\n")

# --- Helper Functions ---

friendly_to_dataset = {
    "hours_studied": "Hours_Studied",
    "attendance": "Attendance",
    "previous_scores": "Previous_Scores",
    "motivation_level": "Motivation_Level",
    "sleep_hours": "Sleep_Hours",
    "internet_access": "Internet_Access",
    "tutoring_sessions": "Tutoring_Sessions",
    "parental_involvement": "Parental_Involvement",
    "teacher_quality": "Teacher_Quality",
    "physical_activity": "Physical_Activity"
}

def clamp_score(v):
    try:
        fv = float(v)
    except:
        return None
    if fv < 0: return 0.0
    if fv > 100: return 100.0
    return fv

def predict_row_friendly(input_dict):
    row = {}
    for col in X_df.columns:
        if col in num_cols:
            row[col] = float(X_df[col].mean())
        else:
            try:
                row[col] = X_df[col].mode().iloc[0]
            except:
                row[col] = X_df[col].iloc[0]
    
    for friendly, val in input_dict.items():
        if friendly not in friendly_to_dataset:
            continue
        ds_col = friendly_to_dataset[friendly]
        if ds_col in num_cols:
            try:
                if ds_col.lower().find('score') >= 0 or ds_col.lower().find('exam') >= 0:
                    clamped = clamp_score(val)
                    row[ds_col] = clamped if clamped is not None else float(X_df[ds_col].mean())
                else:
                    row[ds_col] = float(val)
            except:
                pass
        else:
            row[ds_col] = val
    
    df_row = pd.DataFrame([row], columns=list(X_df.columns))
    Xp = preprocessor.transform(df_row)
    pred = float(model.predict(Xp, verbose=0).flatten()[0])
    pred = max(0.0, min(100.0, float(pred)))
    return round(pred, 2), row

def interpret_grade(g):
    if g >= 90: return {"level": "Excellent", "color": "#28a745", "icon": "bi-star-fill", "explain": "Outstanding (90-100)"}
    if g >= 75: return {"level": "Very Good", "color": "#17a2b8", "icon": "bi-star-half", "explain": "Strong (75-89)"}
    if g >= 50: return {"level": "Needs Improvement", "color": "#ffc107", "icon": "bi-exclamation-triangle", "explain": "Average (50-74)"}
    return {"level": "At Risk", "color": "#dc3545", "icon": "bi-exclamation-octagon", "explain": "Low (<50)"}

def generate_ai_insights(prediction_data):
    insights = []
    hours = prediction_data.get('hours_studied')
    if hours and float(hours) < 5:
        insights.append({"type": "warning", "title": "Increase Study Time", "message": "Try to study more than 5 hours.", "icon": "bi-clock"})
    att = prediction_data.get('attendance')
    if att and float(att) < 75:
        insights.append({"type": "danger", "title": "Low Attendance", "message": "Attendance is critical for success.", "icon": "bi-calendar-x"})
    return insights

# --- Flask Routes ---

@app.route("/")
def home():
    return send_from_directory(str(WEB_DIR), "index.html")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(str(WEB_DIR), filename)

@app.route("/dashboard")
def dashboard_page(): return send_from_directory(str(WEB_DIR), "dashboard.html")

@app.route("/analytics")
def analytics_page(): return send_from_directory(str(WEB_DIR), "analytics.html")

@app.route("/model_visuals")
def model_visuals_page(): return send_from_directory(str(WEB_DIR), "model_visuals.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    print("\n" + "="*40)
    print("📢 NEW PREDICTION REQUEST")
    print(f"📥 Inputs: {json.dumps(data, indent=2)}")
    
    pred, row = predict_row_friendly(data)
    level = interpret_grade(pred)
    insights = generate_ai_insights(data)
    
    print("-" * 20)
    print(f"🤖 Prediction: {pred}%")
    print(f"📊 Level: {level['level']}")
    print("="*40 + "\n")
    
    return jsonify({"ok": True, "pred": pred, "level": level, "insights": insights})

@app.route("/api/model_metrics")
def api_model_metrics():
    # Fetch metrics specifically for Neural Network
    nn = model_performance.get("Neural Network", {})
    
    mae = float(nn.get("MAE", 0))
    mse = float(nn.get("MSE", 0))
    r2  = float(nn.get("R2", 0))
    rmse = float(np.sqrt(mse))
    
    # --- PRINTING TO TERMINAL ---
    print("\n" + "█"*45)
    print("📡 [API] SENDING LIVE MODEL METRICS")
    print("█"*45)
    print(f"   ➤ R² Score (Accuracy): {r2:.4f} ({r2*100:.2f}%)")
    print(f"   ➤ MAE (Mean Abs Error): {mae:.4f}")
    print(f"   ➤ MSE (Mean Sq Error):  {mse:.4f}")
    print(f"   ➤ RMSE (Root MSE):      {rmse:.4f}")
    print("█"*45 + "\n")
    # ----------------------------

    history_path = ARTIFACTS_DIR / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    
    return jsonify({
        "ok": True,
        "metrics": {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        },
        "history": history
    })

@app.route('/api/model_plot_data')
def api_model_plot_data():
    y_true = list(map(float, np.array(y_test).flatten()))
    nn_predictions = model.predict(X_test, verbose=0)
    y_pred = list(map(float, np.array(nn_predictions).flatten()))
    
    feat_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    importances = list(map(float, rf.feature_importances_))
    feature_pairs = [{"feature": f, "importance": v} for f, v in zip(feat_names, importances)]
    feature_pairs = sorted(feature_pairs, key=lambda x: x['importance'], reverse=True)[:10]

    return jsonify({
        "ok": True,
        "predictions_vs_actual": {"y_true": y_true, "y_pred": y_pred},
        "feature_importance": feature_pairs
    })

@app.route("/api/feature_importance")
def api_feature_importance():
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_features)
    importance_scores = rf.feature_importances_
    
    original_feature_importance = {}
    for i, feature in enumerate(feature_names):
        orig = re.sub(r'^(num__|cat__)', '', feature).split('_')[0]
        if orig not in original_feature_importance:
            original_feature_importance[orig] = 0
        original_feature_importance[orig] += importance_scores[i]
        
    total = sum(original_feature_importance.values())
    percents = {k: round((v/total)*100, 2) for k, v in original_feature_importance.items()}
    sorted_feats = dict(sorted(percents.items(), key=lambda x: x[1], reverse=True))
    
    return jsonify({"ok": True, "feature_importance": sorted_feats})

@app.route("/api/dataset_stats")
def api_dataset_stats():
    stats = {
        "total_students": len(df),
        "average_final_score": round(float(df['Exam_Score'].mean()), 2),
        "excellent_students": len(df[df['Exam_Score'] >= 90]),
        "average_hours_studied": round(float(df['Hours_Studied'].mean()), 2)
    }
    return jsonify({"ok": True, "stats": stats})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 SERVER STARTED - http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)