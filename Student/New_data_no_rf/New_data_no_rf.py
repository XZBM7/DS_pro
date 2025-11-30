


# BY XZ - BM7 (ADAPTED FOR NEW DATASET NO RF)



       #####################
     ##                   ##
   ##   ####         ####   ##
  #    #    #       #    #    #
 #     # O  #       #  O #     #
#       ####         ####       #
#                               #
#        \   \_____/   /        #
#         \           /         #
 #         \  -----  /         #
  #          \_____/          #
   ##                       ##
     ##                 ##
       #################

# -----------------------------------------------------------------------------
# STUDENT PERFORMANCE PREDICTOR — FULL SYSTEM DESCRIPTION
# BY XZ - BM7 (Adapted for New Dataset: StudentPerformanceFactors.csv)
# -----------------------------------------------------------------------------
# This project is a complete AI-powered web platform designed to predict student
# exam performance using machine learning, neural networks, a Flask backend,
# JWT authentication, MongoDB storage, analytics, insights, and a full dashboard.
#
# It is a full end-to-end ML system, not a simple training script.
#
# -----------------------------------------------------------------------------
# 1. DATASET & PREPROCESSING
# -----------------------------------------------------------------------------
# - Loads the dataset: StudentPerformanceFactors.csv
# - Removes duplicates and fills missing values
# - Automatically detects:
#       • Numerical columns (scaled with StandardScaler)
#       • Categorical columns (encoded with OneHotEncoder)
# - Builds a ColumnTransformer and saves it using joblib
# - Ensures the same preprocessing is used during training and prediction
#
# -----------------------------------------------------------------------------
# 2. MACHINE LEARNING MODELS
# -----------------------------------------------------------------------------
# The system trains three models:
#   1. Linear Regression          → Baseline model
#   2. RandomForestRegressor      → Strong classical model
#   3. Deep Neural Network (Keras) → Main production model
#
# Neural Network features:
#   - Multiple Dense layers
#   - BatchNormalization
#   - Dropout regularization
#   - Adam optimizer
#   - EarlyStopping + ReduceLROnPlateau
#   - Saved to artifacts/best_model.keras
#
# Models are loaded from disk if available (no retraining needed).
#
# -----------------------------------------------------------------------------
# 3. PREDICTION PIPELINE FLOW
# -----------------------------------------------------------------------------
# 1) Receives “friendly inputs” from frontend (e.g., hours_studied)
# 2) Maps them to dataset columns (Hours_Studied)
# 3) Missing values filled using dataset mean/mode
# 4) Uses saved preprocessor to transform the row
# 5) Neural network predicts final exam score (0–100)
# 6) Prediction is interpreted into:
#       • Excellent
#       • Very Good
#       • Needs Improvement
#       • At Risk
# 7) AI-generated insights created based on inputs (study time, sleep, attendance)
#
# -----------------------------------------------------------------------------
# 4. BACKEND (FLASK)
# -----------------------------------------------------------------------------
# Flask is the central controller that:
#   - Serves API routes
#   - Serves frontend HTML files
#   - Handles authentication
#   - Performs predictions
#   - Computes analytics
#   - Connects to MongoDB
#   - Provides CSV export and metrics endpoints
#
# Protected pages use login_required via JWT.
#
# -----------------------------------------------------------------------------
# 5. AUTHENTICATION (JWT)
# -----------------------------------------------------------------------------
# - Users can register / login
# - Passwords hashed using SHA-256
# - JWT token created and stored in HttpOnly cookies
# - Decorators:
#       • login_required        → Protects frontend pages
#       • api_login_required    → Protects API endpoints
#
# Ensures secure access and prevents unauthorized predictions.
#
# -----------------------------------------------------------------------------
# 6. MONGODB STORAGE
# -----------------------------------------------------------------------------
# MongoDB stores:
#   - Users (full name, username, email, hashed password)
#   - Predictions (score, inputs, level, timestamp)
#
# Unique indexes prevent duplicate accounts.
#
# Prediction documents include:
#   • user_id
#   • pred (0–100)
#   • level (performance category)
#   • friendly input fields
#   • created_at timestamp
#
# -----------------------------------------------------------------------------
# 7. ANALYTICS ENGINE
# -----------------------------------------------------------------------------
# The backend generates statistics for the dashboard:
#   - Average grade
#   - Level distribution (Excellent → At Risk)
#   - Monthly prediction trends
#   - Feature importance (RandomForest)
#   - Neural Network training history
#   - Dataset-level insights (correlations, attendance impact, study hours impact)
#   - Personal recommendations based on user history
#   - Grade improvement plan
#   - Performance benchmarks (user vs dataset)
#
# -----------------------------------------------------------------------------
# 8. FRONTEND INTEGRATION
# -----------------------------------------------------------------------------
# Frontend pages (served from /web) send requests via fetch():
#
#   /api/predict
#   /api/save
#   /api/analytics
#   /api/feature_importance
#   /api/user
#   /api/user/stats
#   /api/personal_recommendations
#
# Pages include:
#   - Dashboard
#   - Insights
#   - Analytics
#   - Records
#   - Profile
#
# -----------------------------------------------------------------------------
# 9. END-TO-END SYSTEM FLOW
# -----------------------------------------------------------------------------
# 1) User logs in → JWT token issued
# 2) User enters prediction data
# 3) Backend maps friendly inputs → dataset columns
# 4) Preprocessor transforms input row
# 5) Neural network predicts final exam score
# 6) Insights + interpretation generated
# 7) Prediction saved in MongoDB
# 8) Dashboard analytics update automatically
#
# Full loop:
#
#   Frontend → Flask → Preprocessor → ML Model → MongoDB → Analytics → Frontend
#
# -----------------------------------------------------------------------------
# 10. SUMMARY
# -----------------------------------------------------------------------------
# This file represents a complete AI platform with:
#   - Machine Learning
#   - Deep Learning
#   - Data preprocessing
#   - API design
#   - Authentication
#   - Database integration
#   - User analytics
#   - Visualization preparation
#
# All modules are connected to form a production-like student performance prediction system.
#
# -----------------------------------------------------------------------------

       #####################
     ##                   ##
   ##   ####         ####   ##
  #    #    #       #    #    #
 #     # O  #       #  O #     #
#       ####         ####       #
#                               #
#        \   \_____/   /        #
#         \           /         #
 #         \  -----  /         #
  #          \_____/          #
   ##                       ##
     ##                 ##
       #################



import os
import sys
import json
import datetime
import csv
import io
import hashlib
import secrets
import re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from flask import Flask, request, jsonify, send_from_directory, Response, redirect, url_for
import jwt
from functools import wraps
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

DATA_FILE = r"D:\Python\DS1\Student\data\StudentPerformanceFactors.csv"
ARTIFACTS_DIR = Path("./artifacts")
WEB_DIR = Path("./web")
MODEL_PATH = ARTIFACTS_DIR / "best_model.keras"
ARTIFACTS_DIR.mkdir(exist_ok=True)
WEB_DIR.mkdir(exist_ok=True)
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "student_predictor_db"
MONGO_COLLECTION = "predictions"
USERS_COLLECTION = "users"

JWT_SECRET = "your-jwt-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

app = Flask(__name__, static_folder=str(WEB_DIR))

THEMES = {
    'light': {
        'bg': '#ffffff',
        'card': '#f8f9fa',
        'text': '#212529',
        'border': '#dee2e6',
        'primary': '#4361ee'
    },
    'dark': {
        'bg': '#1a1a1a',
        'card': '#2d2d2d',
        'text': '#ffffff',
        'border': '#404040',
        'primary': '#4cc9f0'
    }
}

print("Loading dataset...")
if not os.path.exists(DATA_FILE):
    print("ERROR: dataset not found:", DATA_FILE)
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

print("Detected numeric columns:", num_cols)
print("Detected categorical columns:", cat_cols)

from sklearn import __version__ as skl
_major = int(skl.split(".")[0])
_minor = int(skl.split(".")[1]) if len(skl.split("."))>1 else 0
if _major >= 1 and _minor >= 2:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", ohe, cat_cols)])
if PREPROCESSOR_PATH.exists():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Loaded preprocessor from cache")
else:
    preprocessor = preprocessor.fit(X_df)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("Fitted and saved preprocessor")

X_processed = preprocessor.transform(X_df)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model_performance = {}

print("Training classical models (Linear Regression only)...")
lr = LinearRegression().fit(X_train, y_train)

lr_pred = lr.predict(X_test)

model_performance['Linear Regression'] = {
    'MAE': mean_absolute_error(y_test, lr_pred),
    'MSE': mean_squared_error(y_test, lr_pred),
    'R2': r2_score(y_test, lr_pred)
}

print("Classical Models Performance:")
for model_name, metrics in model_performance.items():
    print(f"{model_name}: MAE={metrics['MAE']:.3f}, R²={metrics['R2']:.3f}")

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

    m.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return m

print("Setting up Neural Network...")
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded model from cache:", MODEL_PATH)
else:
    model = build_nn(input_dim)
    es = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    cp = callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), save_best_only=True)

    print("Training Neural Network...")
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

    print("Model trained and saved to:", MODEL_PATH)

nn_preds = model.predict(X_test).flatten()
model_performance['Neural Network'] = {
    'MAE': mean_absolute_error(y_test, nn_preds),
    'MSE': mean_squared_error(y_test, nn_preds),
    'R2': r2_score(y_test, nn_preds)
}

print("Neural Network Performance:")
print(f"MAE: {model_performance['Neural Network']['MAE']:.3f}")
print(f"R²: {model_performance['Neural Network']['R2']:.3f}")

try:
    mongo = MongoClient(MONGO_URI)
    db = mongo[MONGO_DB]
    coll = db[MONGO_COLLECTION]
    users_coll = db[USERS_COLLECTION]
    mongo.admin.command("ping")
    print("MongoDB connected successfully")
    
    users_coll.create_index("email", unique=True)
    users_coll.create_index("username", unique=True)
    
except Exception as e:
    print("MongoDB connection failed:", e)
    coll = None
    users_coll = None


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def generate_jwt_token(user_id, username):
    payload = {
        'user_id': str(user_id),
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS),
        'iat': datetime.datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_current_user():
    token = None
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
    
    if not token:
        token = request.cookies.get('access_token')
    
    if token:
        payload = verify_jwt_token(token)
        if payload:
            return {
                'user_id': payload['user_id'],
                'username': payload['username']
            }
    return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return redirect('/unauthorized')
        return f(*args, **kwargs)
    return decorated_function

def api_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"ok": False, "error": "Authentication required"}), 401
        
        request.user_id = user['user_id']
        request.username = user['username']
        
        return f(*args, **kwargs)
    return decorated_function

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
                    if clamped is None:
                        row[ds_col] = float(X_df[ds_col].mean())
                    else:
                        row[ds_col] = clamped
                else:
                    row[ds_col] = float(val)
            except:
                pass
        else:
            row[ds_col] = val
    
    df_row = pd.DataFrame([row], columns=list(X_df.columns))
    Xp = preprocessor.transform(df_row)
    pred = float(model.predict(Xp).flatten()[0])
    pred = max(0.0, min(100.0, float(pred)))
    return round(pred, 2), row

def interpret_grade(g):
    if g >= 90:
        return {
            "level": "Excellent",
            "color": "#28a745",
            "icon": "bi-star-fill",
            "explain": "Outstanding performance (90-100)"
        }
    if g >= 75:
        return {
            "level": "Very Good",
            "color": "#17a2b8",
            "icon": "bi-star-half",
            "explain": "Strong performance (75-89)"
        }
    if g >= 50:
        return {
            "level": "Needs Improvement",
            "color": "#ffc107",
            "icon": "bi-exclamation-triangle",
            "explain": "Average / needs improvement (50-74)"
        }
    return {
        "level": "At Risk",
        "color": "#dc3545",
        "icon": "bi-exclamation-octagon",
        "explain": "Low performance — support recommended (<50)"
    }

def generate_ai_insights(prediction_data):
    insights = []
    
    hours = prediction_data.get('hours_studied', None)
    try:
        if hours is not None and float(hours) < 5:
            insights.append({
                "type": "warning",
                "title": "Increase Study Time",
                "message": "Studying fewer hours may limit performance. Aim for consistent study hours weekly.",
                "icon": "bi-clock"
            })
    except:
        pass

    sleep = prediction_data.get('sleep_hours', None)
    try:
        if sleep is not None and float(sleep) < 6:
            insights.append({
                "type": "warning",
                "title": "Improve Sleep",
                "message": "Less than 6 hours of sleep can reduce concentration and retention.",
                "icon": "bi-moon"
            })
    except:
        pass

    att = prediction_data.get('attendance', None)
    try:
        if att is not None and float(att) < 75:
            insights.append({
                "type": "danger",
                "title": "Low Attendance",
                "message": f"Attendance {att}% seems low — improving attendance often increases scores.",
                "icon": "bi-calendar-x"
            })
    except:
        pass

    tut = prediction_data.get('tutoring_sessions', 0)
    try:
        if float(tut) >= 1:
            insights.append({
                "type": "info",
                "title": "Tutoring Support",
                "message": "Attending tutoring sessions is correlated with better scores.",
                "icon": "bi-person-bounding-box"
            })
    except:
        pass

    if prediction_data.get('internet_access') in [False, 'no', 'No', 'false', 0, '0']:
        insights.append({
            "type": "secondary",
            "title": "Access to Resources",
            "message": "Limited internet access may reduce ability to use online resources.",
            "icon": "bi-wifi-off"
        })
    
    try:
        mot = prediction_data.get('motivation_level', None)
        if mot is not None and float(mot) < 5:
            insights.append({
                "type": "warning",
                "title": "Motivation Boost",
                "message": "Consider goal-setting and short-term rewards to increase motivation.",
                "icon": "bi-lightbulb"
            })
    except:
        pass

    try:
        tq = prediction_data.get('teacher_quality', None)
        if tq is not None and float(tq) < 3:
            insights.append({
                "type": "info",
                "title": "Teacher Support",
                "message": "Lower teacher quality rating suggests seeking supplemental resources or support.",
                "icon": "bi-person-check"
            })
    except:
        pass

    return insights

@app.route('/api/model_plot_data')
@api_login_required
def api_model_plot_data():
    y_true = []
    y_pred = []
    residuals = []
    feature_pairs = []
    history = None

    try:
        yt = globals().get('y_test')
        Xt = globals().get('X_test')
        if yt is not None and Xt is not None:
            try:
                y_true = list(map(float, np.array(yt).flatten()))
            except:
                y_true = []

            try:
                pred_arr = globals()['model'].predict(Xt)
                y_pred = list(map(float, np.array(pred_arr).flatten()))
            except:
                try:
                    pred_arr = globals()['lr'].predict(Xt)
                    y_pred = list(map(float, np.array(pred_arr).flatten()))
                except:
                    y_pred = []

        if y_true and y_pred and len(y_true) == len(y_pred):
            residuals = [float(t - p) for t, p in zip(y_true, y_pred)]

        lr_model = globals().get('lr')
        pre = globals().get('preprocessor')
        if lr_model is not None and pre is not None:
            try:
                cat_feats = pre.named_transformers_['cat'].get_feature_names_out(cat_cols)
            except:
                try:
                    cat_feats = pre.named_transformers_['cat'].get_feature_names(cat_cols)
                except:
                    cat_feats = []
            
            feat_names = num_cols + list(cat_feats)
            feat_importances = list(map(float, np.abs(lr_model.coef_)))
            
            for f, v in zip(feat_names, feat_importances):
                feature_pairs.append({"feature": f, "importance": v})
            
            feature_pairs = sorted(feature_pairs, key=lambda x: x['importance'], reverse=True)

        hpath = ARTIFACTS_DIR / 'training_history.json'
        if hpath.exists():
            try:
                history = json.loads(hpath.read_text())
            except:
                history = None

        return jsonify({
            "ok": True,
            "predictions_vs_actual": {
                "y_true": y_true,
                "y_pred": y_pred
            },
            "residuals": residuals,
            "feature_importance": feature_pairs,
            "history": history
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
        
@app.route("/model_visuals")
@login_required
def model_visuals_page():
    return send_from_directory(str(WEB_DIR), "model_visuals.html")

def calculate_real_stats(user_id=None):
    if coll is None:
        return None
    
    query = {"user_id": user_id} if user_id else {}
    predictions = list(coll.find(query))
    
    if not predictions:
        return None
    
    pred_values = [p['pred'] for p in predictions]
    level_counts = {}
    for p in predictions:
        level = p.get('level', 'Unknown')
        level_counts[level] = level_counts.get(level, 0) + 1
    
    return {
        "total_predictions": len(predictions),
        "average_grade": round(sum(pred_values) / len(pred_values), 2),
        "level_distribution": level_counts,
        "excellent_count": level_counts.get('Excellent', 0),
        "risk_count": level_counts.get('At Risk', 0)
    }

@app.route("/")
@login_required
def home():
    return send_from_directory(str(WEB_DIR), "index.html")

@app.route("/login")
def login_page():
    if get_current_user():
        return redirect('/')
    return send_from_directory(str(WEB_DIR), "login.html")

@app.route("/register")
def register_page():
    if get_current_user():
        return redirect('/')
    return send_from_directory(str(WEB_DIR), "register.html")

@app.route("/analytics")
@login_required
def analytics_page():
    return send_from_directory(str(WEB_DIR), "analytics.html")

@app.route("/insights")
@login_required
def insights_page():
    return send_from_directory(str(WEB_DIR), "insights.html")

@app.route("/dashboard")
@login_required
def dashboard_page():
    return send_from_directory(str(WEB_DIR), "dashboard.html")

@app.route("/guide")
@login_required
def guide_page():
    return send_from_directory(str(WEB_DIR), "guide.html")

@app.route("/unauthorized")
def unauthorized_page():
    return send_from_directory(str(WEB_DIR), "unauthorized.html")

@app.route("/api/register", methods=["POST"])
def api_register():
    if users_coll is None:
        return jsonify({"ok": False, "error": "Database not connected"})
    
    data = request.get_json() or {}
    full_name = data.get('full_name', '').strip()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    confirm_password = data.get('confirm_password', '')
    
    if not all([full_name, username, email, password, confirm_password]):
        return jsonify({"ok": False, "error": "All fields are required"})
    
    if len(full_name) < 2:
        return jsonify({"ok": False, "error": "Full name must be at least 2 characters long"})
    
    if not re.match(r'^[a-zA-Z\u0600-\u06FF\s]+$', full_name):
        return jsonify({"ok": False, "error": "Full name can only contain letters and spaces"} )
    
    if len(username) < 3:
        return jsonify({"ok": False, "error": "Username must be at least 3 characters long"})
    
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return jsonify({"ok": False, "error": "Username can only contain letters, numbers, and underscores"})
    
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return jsonify({"ok": False, "error": "Please enter a valid email address"})
    
    if len(password) < 6:
        return jsonify({"ok": False, "error": "Password must be at least 6 characters long"})
    
    if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)', password):
        return jsonify({"ok": False, "error": "Password must contain at least one lowercase letter, one uppercase letter, and one number"})
    
    if password != confirm_password:
        return jsonify({"ok": False, "error": "Passwords do not match"})
    
    if users_coll.find_one({"$or": [{"email": email}, {"username": username}]}):
        return jsonify({"ok": False, "error": "User already exists with this email or username"})
    
    user_data = {
        "full_name": full_name,
        "username": username,
        "email": email,
        "password": hash_password(password),
        "created_at": datetime.datetime.utcnow(),
        "last_login": None
    }
    
    try:
        result = users_coll.insert_one(user_data)
        return jsonify({"ok": True, "message": "User created successfully"})
    except Exception as e:
        return jsonify({"ok": False, "error": "Registration failed"})

@app.route("/api/login", methods=["POST"])
def api_login():
    if users_coll is None:
        return jsonify({"ok": False, "error": "Database not connected"})
    
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({"ok": False, "error": "Username/email and password are required"})
    
    if len(username) < 3:
        return jsonify({"ok": False, "error": "Username/email must be at least 3 characters long"})
    
    if '@' in username:
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', username):
            return jsonify({"ok": False, "error": "Please enter a valid email address"})
    else:
        if not re.match(r'^[a-zA-Z0-9_]{3,}$', username):
            return jsonify({"ok": False, "error": "Username can only contain letters, numbers, and underscores"})
    
    if len(password) < 1:
        return jsonify({"ok": False, "error": "Password is required"})
    
    user = users_coll.find_one({
        "$or": [
            {"email": username.lower()},
            {"username": username}
        ]
    })
    
    if not user or not verify_password(password, user['password']):
        return jsonify({"ok": False, "error": "Invalid username/email or password"})
    
    if user.get('status') == 'inactive':
        return jsonify({"ok": False, "error": "Your account has been deactivated. Please contact support."})
    
    users_coll.update_one(
        {"_id": user['_id']},
        {"$set": {"last_login": datetime.datetime.utcnow()}}
    )
    
    token = generate_jwt_token(user['_id'], user['username'])
    
    response = jsonify({
        "ok": True,
        "message": "Login successful",
        "user": {
            "id": str(user['_id']),
            "full_name": user.get('full_name'),
            "username": user.get('username'),
            "email": user.get('email')
        }
    })
    
    response.set_cookie(
        'access_token',
        token,
        httponly=True,
        secure=False,
        samesite='Lax',
        max_age=JWT_EXPIRY_HOURS * 3600
    )
    
    return response

@app.route("/api/logout", methods=["POST"])
def api_logout():
    response = jsonify({"ok": True, "message": "Logged out successfully"})
    response.set_cookie('access_token', '', expires=0)
    return response

@app.route("/api/user")
def api_user():
    user = get_current_user()
    if user and users_coll is not None:
        db_user = users_coll.find_one({"_id": ObjectId(user['user_id'])})
        if db_user:
            return jsonify({
                "ok": True,
                "user": {
                    "id": str(db_user['_id']),
                    "full_name": db_user.get('full_name'),
                    "username": db_user.get('username'),
                    "email": db_user.get('email')
                }
            })
    return jsonify({"ok": False, "error": "Not authenticated"})

@app.route("/api/user/stats")
@api_login_required
def api_user_stats():
    stats = calculate_real_stats(request.user_id)
    if stats:
        return jsonify({"ok": True, "stats": stats})
    return jsonify({"ok": True, "stats": {
        "total_predictions": 0,
        "average_grade": 0.0,
        "level_distribution": {},
        "excellent_count": 0,
        "risk_count": 0
    }})

@app.route("/api/predict", methods=["POST"])
@api_login_required
def api_predict():
    data = request.get_json() or {}
    pred, row = predict_row_friendly(data)
    level = interpret_grade(pred)
    insights = generate_ai_insights(data)
    return jsonify({"ok": True, "pred": pred, "level": level, "insights": insights})

@app.route("/api/save", methods=["POST"])
@api_login_required
def save_prediction():
    try:
        if coll is None:
            return jsonify({"ok": False, "error": "MongoDB not connected"})
        
        if not request.is_json:
            return jsonify({"ok": False, "error": "Request must be JSON"})
        
        payload = request.get_json()
        if not payload:
            return jsonify({"ok": False, "error": "No data provided"})
        
        user_id = request.user_id
        
        fields = payload.get("fields") or payload.get("inputs") or {}
        pred = payload.get("pred")
        level = payload.get("level", "")
        
        if pred is None:
            return jsonify({"ok": False, "error": "Prediction value is required"})
        
        try:
            pred = float(pred)
        except (ValueError, TypeError):
            return jsonify({"ok": False, "error": "Invalid prediction value"})
        
        user = users_coll.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"ok": False, "error": "User not found"})
        
        doc = {
            "user_id": user_id,
            "user_name": user.get('full_name') or user.get('username'),
            "pred": pred,
            "level": level,
            "created_at": datetime.datetime.utcnow()
        }
        
        for friendly, ds_col in friendly_to_dataset.items():
            val = fields.get(friendly, None)
            doc[friendly] = val
        
       
        res = coll.insert_one(doc)
        return jsonify({
            "ok": True,
            "id": str(res.inserted_id),
            "message": "Prediction saved successfully"
        })
        
    except Exception as e:
        print(f"Save prediction error: {e}")
        return jsonify({"ok": False, "error": "Server error: " + str(e)})

@app.route("/api/records")
@api_login_required
def get_records():
    if coll is None:
        return jsonify({"ok": False, "error": "MongoDB not connected"})
    
    user_id = request.user_id
    rows = []
    for d in coll.find({"user_id": user_id}).sort("created_at", -1):
        rows.append({
            "_id": str(d.get("_id")),
            "pred": d.get("pred"),
            "level": d.get("level"),
            "created_at": d.get("created_at").strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify({"ok": True, "rows": rows})

@app.route("/api/delete/<rid>", methods=["DELETE"])
@api_login_required
def delete_record(rid):
    if coll is None:
        return jsonify({"ok": False, "error": "MongoDB not connected"})
    
    user_id = request.user_id
    result = coll.delete_one({"_id": ObjectId(rid), "user_id": user_id})
    
    if result.deleted_count > 0:
        return jsonify({"ok": True})
    else:
        return jsonify({"ok": False, "error": "Prediction not found or access denied"})

@app.route("/api/export_csv")
@api_login_required
def export_csv():
    if coll is None:
        return jsonify({"ok": False, "error": "MongoDB not connected"})
    
    user_id = request.user_id
    rows = list(coll.find({"user_id": user_id}).sort("created_at", -1))
    si = io.StringIO()
    writer = csv.writer(si)
    header = ["id"] + list(friendly_to_dataset.keys()) + ["pred","level","created_at"]
    writer.writerow(header)
    for d in rows:
        row = [
            str(d.get("_id"))
        ]
        for friendly in friendly_to_dataset.keys():
            row.append(d.get(friendly))
        row.extend([
            d.get("pred"),
            d.get("level"),
            d.get("created_at").strftime("%Y-%m-%d %H:%M:%S")
        ])
        writer.writerow(row)
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=my_predictions.csv"}
    )

@app.route("/api/analytics")
@api_login_required
def api_analytics():
    if coll is None:
        return jsonify({"ok": False, "error": "MongoDB not connected"})
    
    user_id = request.user_id
    
    predictions = list(coll.find(
        {"user_id": user_id},
        {'pred': 1, 'level': 1, 'created_at': 1}
    ))
    
    if not predictions:
        return jsonify({"ok": False, "error": "No data available"})
    
    pred_values = [p['pred'] for p in predictions]
    level_counts = {}
    for p in predictions:
        level = p.get('level', 'Unknown')
        level_counts[level] = level_counts.get(level, 0) + 1

    df_pred = pd.DataFrame(predictions)
    df_pred['created_at'] = pd.to_datetime(df_pred['created_at'])
    df_pred['month'] = df_pred['created_at'].dt.strftime('%Y-%m')

    trend_df = df_pred.groupby('month')['pred'].mean().reset_index()

    prediction_trends = {
        "months": trend_df['month'].tolist(),
        "averages": trend_df['pred'].round(2).tolist()
    }

    return jsonify({
        "ok": True,
        "stats": {
            "total_predictions": len(predictions),
            "average_grade": round(sum(pred_values) / len(pred_values), 2),
            "level_distribution": level_counts,
            "model_performance": model_performance,
            "prediction_trends": prediction_trends 
        }
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(str(WEB_DIR), filename)

@app.route("/api/health")
def health_check():
    return jsonify({
        "ok": True,
        "message": "Server is running",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "mongo_connected": coll is not None,
        "model_loaded": model is not None
    })

@app.route("/api/refresh", methods=["POST"])
def refresh_token():
    user = get_current_user()
    if not user:
        return jsonify({"ok": False, "error": "Invalid token"}), 401
    
    token = generate_jwt_token(user['user_id'], user['username'])
    
    response = jsonify({"ok": True, "message": "Token refreshed"})
    response.set_cookie(
        'access_token',
        token,
        httponly=True,
        secure=False,
        samesite='Lax',
        max_age=JWT_EXPIRY_HOURS * 3600
    )
    
    return response

@app.route("/api/dataset_stats")
@api_login_required
def api_dataset_stats():
    try:
        correlations = {}
        if 'Previous_Scores' in df.columns and 'Exam_Score' in df.columns:
            correlations['Previous_Scores'] = float(df['Previous_Scores'].corr(df['Exam_Score']))
        if 'Hours_Studied' in df.columns and 'Exam_Score' in df.columns:
            correlations['Hours_Studied'] = float(df['Hours_Studied'].corr(df['Exam_Score']))
        
        study_time_impact = {}
        if 'Hours_Studied' in df.columns and 'Exam_Score' in df.columns:
            for hrs in sorted(df['Hours_Studied'].unique()):
                avg_grade = float(df[df['Hours_Studied'] == hrs]['Exam_Score'].mean())
                study_time_impact[float(hrs)] = round(avg_grade, 2)
        
        absence_impact = {}
        if 'Attendance' in df.columns and 'Exam_Score' in df.columns:
            try:
                df['attendance_group'] = pd.cut(df['Attendance'], bins=[0,50,70,85,100], labels=['0-50','51-70','71-85','86-100'])
                for group in ['0-50','51-70','71-85','86-100']:
                    group_data = df[df['attendance_group'] == group]
                    if len(group_data) > 0:
                        absence_impact[group] = round(float(group_data['Exam_Score'].mean()), 2)
            except:
                pass
        
        tutoring_impact = {}
        if 'Tutoring_Sessions' in df.columns and 'Exam_Score' in df.columns:
            for t in sorted(df['Tutoring_Sessions'].unique()):
                group = df[df['Tutoring_Sessions'] == t]
                if len(group) > 0:
                    tutoring_impact[int(t)] = round(float(group['Exam_Score'].mean()), 2)
        
        stats = {
            "total_students": len(df),
            "average_final_score": round(float(df['Exam_Score'].mean()), 2),
            "excellent_students": len(df[df['Exam_Score'] >= 90]),
            "at_risk_students": len(df[df['Exam_Score'] < 50]),
            "average_hours_studied": round(float(df['Hours_Studied'].mean()), 2),
            "average_attendance": round(float(df['Attendance'].mean()), 2)
        }
        
        return jsonify({
            "ok": True,
            "correlations": correlations,
            "study_time_impact": study_time_impact,
            "attendance_impact": absence_impact,
            "tutoring_impact": tutoring_impact,
            "stats": stats
        })
        
    except Exception as e:
        print(f"Error in dataset stats: {e}")
        return jsonify({"ok": False, "error": "Failed to analyze dataset"})

@app.route("/api/feature_importance")
@api_login_required
def api_feature_importance():
    try:
        try:
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        except:
            try:
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names(cat_cols)
            except:
                cat_features = []
        feature_names = num_cols + list(cat_features)
        
        importance_scores = np.abs(lr.coef_)
        
        original_feature_importance = {}
        
        for i, feature in enumerate(feature_names):
            original_feature = feature
            original_feature = re.sub(r'^(num__|cat__)', '', original_feature)
            original_feature = re.split(r'[_\-]', original_feature)[0]
            if original_feature not in original_feature_importance:
                original_feature_importance[original_feature] = 0
            original_feature_importance[original_feature] += importance_scores[i]
        
        total_importance = sum(original_feature_importance.values()) if original_feature_importance else 1
        feature_importance_percent = {}
        
        for feature, importance in original_feature_importance.items():
            feature_importance_percent[feature] = round((importance / total_importance) * 100, 2)
        
        sorted_features = dict(sorted(
            feature_importance_percent.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return jsonify({
            "ok": True,
            "feature_importance": sorted_features
        })
        
    except Exception as e:
        print(f"Error in feature importance: {e}")
        return jsonify({"ok": False, "error": "Failed to calculate feature importance"})

@app.route("/api/personal_recommendations")
@api_login_required
def api_personal_recommendations():
    try:
        user_id = request.user_id
        
        recent_predictions = list(coll.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(5))
        
        if not recent_predictions:
            return jsonify({
                "ok": True,
                "recommendations": get_general_recommendations()
            })
        
        predictions_df = pd.DataFrame(recent_predictions)
        
        recommendations = []
        
        avg_hours = float(predictions_df.get('hours_studied', pd.Series([0])).mean())
        if avg_hours < 5:
            recommendations.append({
                "type": "study_time",
                "priority": "high",
                "title": "Increase Study Time",
                "message": f"Current study time ({avg_hours:.1f} hours) is below recommended. Aim for regular study sessions.",
                "action": "Create a regular study schedule",
                "icon": "bi-clock"
            })
        
        avg_att = float(predictions_df.get('attendance', pd.Series([100])).mean())
        if avg_att < 75:
            recommendations.append({
                "type": "attendance",
                "priority": "high",
                "title": "Improve Attendance",
                "message": f"Average attendance ({avg_att:.1f}%) is low. Regular attendance can improve performance.",
                "action": "Identify reasons for absence and try to reduce them",
                "icon": "bi-calendar-check"
            })
        
        if len(recommendations) < 3:
            recommendations.extend(get_general_recommendations())
        
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])
        
        return jsonify({
            "ok": True,
            "recommendations": recommendations[:5]
        })
        
    except Exception as e:
        print(f"Error in personal recommendations: {e}")
        return jsonify({
            "ok": True,
            "recommendations": get_general_recommendations()
        })

def get_general_recommendations():
    return [
        {
            "type": "study_habits",
            "priority": "medium",
            "title": "Optimize Study Environment",
            "message": "Dedicated study spaces and focused sessions increase retention.",
            "action": "Create a quiet, organized study area",
            "icon": "bi-house"
        },
        {
            "type": "peer_learning",
            "priority": "medium",
            "title": "Join Study Groups",
            "message": "Collaborative learning improves understanding and retention.",
            "action": "Form or join a study group",
            "icon": "bi-people"
        },
        {
            "type": "health",
            "priority": "low",
            "title": "Balance Work and Rest",
            "message": "Adequate sleep and breaks improve learning efficiency.",
            "action": "Ensure 7-8 hours of sleep nightly",
            "icon": "bi-heart"
        }
    ]

@app.route("/api/grade_improvement_plan")
@api_login_required
def api_grade_improvement_plan():
    try:
        user_id = request.user_id
        
        latest_prediction = coll.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )
        
        if not latest_prediction:
            return jsonify({
                "ok": True,
                "plan": get_default_improvement_plan()
            })
        
        current_score = latest_prediction.get('pred', 50)
        hours = latest_prediction.get('hours_studied', 0)
        attendance = latest_prediction.get('attendance', 100)
        tutoring = latest_prediction.get('tutoring_sessions', 0)
        
        improvement_plan = {
            "current_score": current_score,
            "target_score": min(100, current_score + 8),
            "timeframe": "4-8 weeks",
            "actions": []
        }
        
        if hours < 5:
            improvement_plan["actions"].append({
                "area": "Study Time",
                "action": f"Increase from {hours} to {min(10, hours + 3)} hours weekly",
                "impact": "Expected +3 to +5 points",
                "timeline": "Immediate"
            })
        
        if attendance < 80:
            improvement_plan["actions"].append({
                "area": "Attendance",
                "action": f"Improve attendance from {attendance}% to {min(100, attendance + 10)}%",
                "impact": "Expected +2 to +4 points",
                "timeline": "2-4 weeks"
            })
        
        if tutoring == 0:
            improvement_plan["actions"].append({
                "area": "Tutoring",
                "action": "Attend targeted tutoring sessions for weak topics",
                "impact": "Expected +3 points",
                "timeline": "3-6 weeks"
            })
        
        improvement_plan["actions"].append({
            "area": "Exam Strategy",
            "action": "Practice with past exam papers weekly",
            "impact": "Expected +2 to +4 points",
            "timeline": "4 weeks"
        })
        
        return jsonify({
            "ok": True,
            "plan": improvement_plan
        })
        
    except Exception as e:
        print(f"Error in improvement plan: {e}")
        return jsonify({
            "ok": True,
            "plan": get_default_improvement_plan()
        })

def get_default_improvement_plan():
    return {
        "current_score": 50,
        "target_score": 58,
        "timeframe": "4-6 weeks",
        "actions": [
            {
                "area": "Study Time",
                "action": "Increase to 6-8 hours weekly",
                "impact": "Expected +3 points",
                "timeline": "Immediate"
            },
            {
                "area": "Study Methods",
                "action": "Implement active recall techniques",
                "impact": "Expected +2 points",
                "timeline": "2 weeks"
            },
            {
                "area": "Exam Preparation",
                "action": "Weekly practice tests",
                "impact": "Expected +3 points",
                "timeline": "4 weeks"
            }
        ]
    }

@app.route("/api/performance_benchmarks")
@api_login_required
def api_performance_benchmarks():
    try:
        user_id = request.user_id
        
        user_predictions = list(coll.find({"user_id": user_id}))
        
        if not user_predictions:
            return jsonify({
                "ok": True,
                "benchmarks": get_general_benchmarks()
            })
        
        user_df = pd.DataFrame(user_predictions)
        
        benchmarks = {
            "user_stats": {
                "average_score": round(float(user_df['pred'].mean()), 2),
                "total_predictions": len(user_predictions),
                "best_score": float(user_df['pred'].max()),
                "improvement_trend": calculate_improvement_trend(user_predictions)
            },
            "dataset_benchmarks": {
                "average_score": round(float(df['Exam_Score'].mean()), 2),
                "top_25_percent": round(float(df['Exam_Score'].quantile(0.75)), 2),
                "excellent_threshold": 90.0,
                "at_risk_threshold": 50.0
            },
            "comparison": compare_with_benchmarks(user_df, df)
        }
        
        return jsonify({
            "ok": True,
            "benchmarks": benchmarks
        })
        
    except Exception as e:
        print(f"Error in performance benchmarks: {e}")
        return jsonify({
            "ok": True,
            "benchmarks": get_general_benchmarks()
        })

def calculate_improvement_trend(predictions):
    if len(predictions) < 3:
        return "insufficient_data"
    
    sorted_predictions = sorted(predictions, key=lambda x: x['created_at'])
    scores = [p['pred'] for p in sorted_predictions]
    
    if scores[-1] > scores[0]:
        return "improving"
    elif scores[-1] < scores[0]:
        return "declining"
    else:
        return "stable"

def compare_with_benchmarks(user_df, dataset_df):
    user_avg = user_df['pred'].mean()
    dataset_avg = dataset_df['Exam_Score'].mean()
    
    if user_avg > dataset_avg:
        return f"Above average by {user_avg - dataset_avg:.1f} points"
    elif user_avg < dataset_avg:
        return f"Below average by {dataset_avg - user_avg:.1f} points"
    else:
        return "At average level"

def get_general_benchmarks():
    return {
        "user_stats": {
            "average_score": 0,
            "total_predictions": 0,
            "best_score": 0,
            "improvement_trend": "no_data"
        },
        "dataset_benchmarks": {
            "average_score": round(float(df['Exam_Score'].mean()), 2),
            "top_25_percent": round(float(df['Exam_Score'].quantile(0.75)), 2),
            "excellent_threshold": 90.0,
            "at_risk_threshold": 50.0
        },
        "comparison": "No user data available for comparison"
    }

@app.route("/records")
@login_required
def records_page():
    return send_from_directory(str(WEB_DIR), "records.html")

@app.route("/profile")
@login_required
def profile_page():
    return send_from_directory(str(WEB_DIR), "profile.html")

@app.route("/api/user/profile", methods=["GET"])
@api_login_required
def get_user_profile():
    try:
        user_id = request.user_id
        user = users_coll.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            return jsonify({"ok": False, "error": "User not found"})
        
        profile_data = {
            "full_name": user.get('full_name', ''),
            "username": user.get('username', ''),
            "email": user.get('email', ''),
            "created_at": user.get('created_at').strftime("%Y-%m-%d %H:%M:%S") if user.get('created_at') else "Unknown",
            "last_login": user.get('last_login').strftime("%Y-%m-%d %H:%M:%S") if user.get('last_login') else "Never"
        }
        
        return jsonify({"ok": True, "profile": profile_data})
        
    except Exception as e:
        print(f"Error getting profile: {e}")
        return jsonify({"ok": False, "error": "Failed to get profile data"})

@app.route("/api/user/update", methods=["PUT"])
@api_login_required
def update_user_profile():
    try:
        if users_coll is None:
            return jsonify({"ok": False, "error": "Database not connected"})
        
        data = request.get_json() or {}
        user_id = request.user_id
        
        full_name = data.get('full_name', '').strip()
        username = data.get('username', '').strip()
        
        if not full_name or not username:
            return jsonify({"ok": False, "error": "All fields are required"})
        
        if len(full_name) < 2:
            return jsonify({"ok": False, "error": "Full name must be at least 2 characters long"})
        
        if not re.match(r'^[a-zA-Z\u0600-\u06FF\s]+$', full_name):
            return jsonify({"ok": False, "error": "Full name can only contain letters and spaces"})
        
        if len(username) < 3:
            return jsonify({"ok": False, "error": "Username must be at least 3 characters long"})
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return jsonify({"ok": False, "error": "Username can only contain letters, numbers, and underscores"})
        
        existing_user = users_coll.find_one({
            "username": username,
            "_id": {"$ne": ObjectId(user_id)}
        })
        
        if existing_user:
            return jsonify({"ok": False, "error": "Username already taken"})
        
        update_data = {
            "full_name": full_name,
            "username": username,
            "updated_at": datetime.datetime.utcnow()
        }
        
        result = users_coll.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return jsonify({"ok": True, "message": "Profile updated successfully"})
        else:
            return jsonify({"ok": True, "message": "No changes made"})
            
    except Exception as e:
        print(f"Error updating profile: {e}")
        return jsonify({"ok": False, "error": "Failed to update profile"})

@app.route("/api/user/change-password", methods=["PUT"])
@api_login_required
def change_password():
    try:
        if users_coll is None:
            return jsonify({"ok": False, "error": "Database not connected"})
        
        data = request.get_json() or {}
        user_id = request.user_id
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        confirm_password = data.get('confirm_password', '')
        
        if not current_password or not new_password or not confirm_password:
            return jsonify({"ok": False, "error": "All password fields are required"})
        
        user = users_coll.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"ok": False, "error": "User not found"})
        
        if not verify_password(current_password, user['password']):
            return jsonify({"ok": False, "error": "Current password is incorrect"})
        
        if len(new_password) < 6:
            return jsonify({"ok": False, "error": "New password must be at least 6 characters long"})
        
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)', new_password):
            return jsonify({"ok": False, "error": "New password must contain at least one lowercase letter, one uppercase letter, and one number"})
        
        if new_password != confirm_password:
            return jsonify({"ok": False, "error": "New passwords do not match"})
        
        if verify_password(new_password, user['password']):
            return jsonify({"ok": False, "error": "New password must be different from current password"})
        
        hashed_new_password = hash_password(new_password)
        
        result = users_coll.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "password": hashed_new_password,
                "updated_at": datetime.datetime.utcnow()
            }}
        )
        
        if result.modified_count > 0:
            return jsonify({"ok": True, "message": "Password changed successfully"})
        else:
            return jsonify({"ok": False, "error": "Failed to change password"})
            
    except Exception as e:
        print(f"Error changing password: {e}")
        return jsonify({"ok": False, "error": "Failed to change password"})

@app.route("/api/user/activity")
@api_login_required
def get_user_activity():
    try:
        user_id = request.user_id
        
        prediction_stats = calculate_real_stats(user_id)
        
        first_prediction = coll.find_one(
            {"user_id": user_id},
            sort=[("created_at", 1)]
        )
        last_prediction = coll.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )
        
        activity_data = {
            "prediction_stats": prediction_stats or {
                "total_predictions": 0,
                "average_grade": 0.0,
                "level_distribution": {},
                "excellent_count": 0,
                "risk_count": 0
            },
            "first_prediction_date": first_prediction.get('created_at').strftime("%Y-%m-%d") if first_prediction else "No predictions yet",
            "last_prediction_date": last_prediction.get('created_at').strftime("%Y-%m-%d") if last_prediction else "No predictions yet",
            "prediction_frequency": calculate_prediction_frequency(user_id)
        }
        
        return jsonify({"ok": True, "activity": activity_data})
        
    except Exception as e:
        print(f"Error getting activity: {e}")
        return jsonify({"ok": False, "error": "Failed to get activity data"})

def calculate_prediction_frequency(user_id):
    predictions = list(coll.find(
        {"user_id": user_id},
        {'created_at': 1}
    ).sort("created_at", 1))
    
    if len(predictions) < 2:
        return "Not enough data"
    
    first_date = predictions[0]['created_at']
    last_date = predictions[-1]['created_at']
    total_days = (last_date - first_date).days + 1
    
    if total_days <= 0:
        return "Daily"
    
    frequency = len(predictions) / total_days
    
    if frequency >= 1:
        return "Daily"
    elif frequency >= 0.5:
        return "Every 2 days"
    elif frequency >= 0.14:
        return "Weekly"
    else:
        return "Monthly"

@app.route("/api/model_metrics")
def api_model_metrics():
    try:
        nn = model_performance.get("Neural Network", {})
        lr = model_performance.get("Linear Regression", {})
        
        
        mae = nn.get("MAE", 0)
        mse = nn.get("MSE", 0)
        r2  = nn.get("R2", 0)
        rmse = float(np.sqrt(mse)) if mse else 0

        history_path = ARTIFACTS_DIR / "training_history.json"
        history = {}
        loss = None
        val_loss = None

        if history_path.exists():
            history = json.loads(history_path.read_text())
            loss = history.get("loss", [])
            val_loss = history.get("val_loss", [])

        last_loss = float(loss[-1]) if loss else None
        last_val_loss = float(val_loss[-1]) if val_loss else None

        return jsonify({
            "ok": True,
            "metrics": {
                "MAE": round(float(mae), 3),
                "MSE": round(float(mse), 3),
                "RMSE": round(float(rmse), 3),
                "R2": round(float(r2), 3),
                "Loss": round(last_loss, 3) if last_loss else None,
                "Val_Loss": round(last_val_loss, 3) if last_val_loss else None,
            },
            "history": history
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    print("🚀 Student Performance Predictor Server Starting...")
    print("🔐 JWT Authentication Enabled")
    print("📊 Model Performance Summary (NN & LR only):")
    for model_name, metrics in model_performance.items():
        print(f"   {model_name}: R² = {metrics['R2']:.3f}")
    
    print("\n🌐 Server running → http://127.0.0.1:5000/")
    print("📁 Frontend pages served from:", WEB_DIR.absolute())
    
    app.run(host="127.0.0.1", port=5000, debug=False)