🎓 Student Performance Predictor System

A full-stack Machine Learning–powered web application designed to predict students’ exam scores, provide intelligent insights, and visualize academic performance trends using historical data.

📌 Project Overview

The Student Performance Predictor System helps students estimate their exam scores based on multiple academic and personal factors such as study hours, attendance, sleep, motivation, and tutoring sessions.

The system combines:

Machine Learning models

Flask backend APIs

MongoDB database

Secure JWT authentication

Interactive analytics dashboards

This project is suitable for academic submission, graduation projects, and ML-based system demonstrations.

🚀 Features
🔐 Authentication & Security

User registration and login

JWT-based authentication

Secure password hashing

Protected API routes

🤖 Machine Learning

Neural Network model (TensorFlow)

Linear Regression model (Scikit-learn)

Automated preprocessing (scaling & encoding)

Model performance evaluation (MAE, MSE, RMSE, R²)

📊 Prediction & Insights

Predict exam scores (0–100)

Performance level interpretation

AI-generated insights and recommendations

Personalized improvement plans

📈 Analytics & Visualization

Prediction history tracking

Performance trends over time

Dataset statistics and correlations

Interactive charts (Plotly)

🗂 Data Management

Save and manage predictions

Delete prediction records

Export predictions to CSV

User activity tracking

👤 Profile Management

View & update profile

Change password securely

🏗️ System Architecture

The system follows a Layered Architecture:

Client (Browser)
   ↓
Frontend UI (HTML / CSS / JS)
   ↓
Flask Backend API
   ↓
Machine Learning Engine
   ↓
MongoDB Database

🧰 Technology Stack
Layer	Technology
Frontend	HTML, CSS, JavaScript
Backend	Python, Flask
ML	TensorFlow, Scikit-learn
Data	MongoDB
Auth	JWT
Visualization	Plotly
Storage	Joblib, Keras
📁 Project Structure
project/
│
├── app.py
├── artifacts/
│   ├── best_model.keras
│   ├── preprocessor.pkl
│   └── training_history.json
│
├── web/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── analytics.html
│   ├── records.html
│   └── profile.html
│
├── data/
│   └── StudentPerformanceFactors.csv

⚙️ System Requirements
Software

Python 3.9 – 3.11

MongoDB 5.0+

Modern web browser (Chrome recommended)

Python Libraries
pip install numpy pandas scikit-learn tensorflow joblib flask pyjwt pymongo plotly

▶️ How to Run the Project
1️⃣ Start MongoDB
mongod

2️⃣ Set JWT Secret

In app.py:

JWT_SECRET = "your_secret_key_here"

3️⃣ Run the Flask Server
python app.py

4️⃣ Open the Application
http://127.0.0.1:5000

📊 Machine Learning Pipeline

Load dataset (StudentPerformanceFactors.csv)

Clean and preprocess data

Apply feature scaling & encoding

Train ML models (NN + LR)

Evaluate performance

Save trained models as artifacts

Use trained model for real-time predictions

🔐 Security Notes

JWT tokens stored in HTTP-only cookies

Passwords are hashed before storage

All sensitive endpoints are protected

🧪 Testing

Manual API testing

Model performance validation

Authentication flow testing

Prediction accuracy verification

🚧 Future Enhancements

Cloud deployment (AWS / Azure)

Admin dashboard

Automatic model retraining

Role-based access control

Multi-dataset support

📄 Documentation

The project includes:

Use Case Diagram

Sequence Diagrams

Activity Diagrams

DFD (Level 0 & 1)

Block Diagram

Package & Component Diagrams

Full System Architecture Diagram

👨‍💻 Author

Ibrahim Amr
Student & Software Developer
Specialized in Databases, Web Development, and Machine Learning

📜 License

This project is developed for educational and academic purposes.

⭐ Final Note

If you find this project useful, feel free to ⭐ star the repository or use it as a reference for your own academic projects.
