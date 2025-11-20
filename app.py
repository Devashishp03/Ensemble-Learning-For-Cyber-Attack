from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import smtplib
import random
import requests  
from flask_wtf.csrf import CSRFProtect
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from joblib import dump, load
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from threading import Lock
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import traceback
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import datetime
import pyshark
from threading import Thread
from queue import Queue
import psutil
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
csrf = CSRFProtect(app)

# ====================== Database & Config ======================
def init_db():
    conn = sqlite3.connect('registration.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_registration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Username TEXT NOT NULL,
            Email TEXT NOT NULL,
            Password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

EMAIL_CONFIG = {
    'sender_email': "ensemblecyberdetection@gmail.com",
    'sender_password': "bbccqzgtohhcwrar",
    'smtp_server': "smtp.gmail.com",
    'smtp_port': 587
}

# ====================== Dataset Configuration ======================
SELECTED_FEATURES = [
    'Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Down_Up_Ratio', 'Average_Packet_Size', 'act_data_pkt_fwd',
    'min_seg_size_forward'
]

ATTACK_TYPES = {
    0: 'Normal',
    1: 'DDOS Attack',
    2: 'PortScan Attack',
    3: 'Bot Attack',
    4: 'Web Attack',
    5: 'Brute Force Attack'
}

# ====================== Global Variables ======================
attack_history = []
model_accuracies = {}
traffic_queue = Queue()
is_capturing = False
realtime_detections = []
training_lock = Lock()
MAX_REALTIME_DETECTIONS = 50  # Keep last 50 detections
DATASET_PATH = 'Cyber Attack.csv'  

# ====================== Core Functions ======================
def load_and_preprocess_data():
    """Improved data loading with validation"""
    try:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
            
        data = pd.read_csv(DATASET_PATH)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Validate required columns exist
        required_columns = SELECTED_FEATURES + ['Label']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return data
    except Exception as e:
        app.logger.error(f"Data loading error: {str(e)}\n{traceback.format_exc()}")
        raise

def train_model(model_name, dataset=None):
    """Improved model training with complete error handling and file upload support."""
    try:
        with training_lock:  # Lock entire training process
            # Data loading and validation
            if dataset is None:
                data = load_and_preprocess_data()
            else:
                data = dataset.replace([np.inf, -np.inf], np.nan).dropna()

                # Validate required columns exist
                required_columns = SELECTED_FEATURES + ['Label']
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")

            if len(data) < 100:
                raise ValueError("Insufficient data (min 100 samples required)")

            X = data[SELECTED_FEATURES]
            y = data['Label']

            if len(y.unique()) < 2:
                raise ValueError("Need at least 2 classes in training data")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)

            k_features = min(7, X_train.shape[1])
            selector = SelectKBest(f_classif, k=k_features)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

            model_config = {
                "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced'),
                "RandomForest": RandomForestClassifier(n_estimators=150, class_weight='balanced', n_jobs=-1),
                "NaiveBayes": GaussianNB(),
                "Ensemble": VotingClassifier(
                    estimators=[
                        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced')),
                        ('rf', RandomForestClassifier(n_estimators=150, class_weight='balanced')),
                        ('nb', GaussianNB())
                    ],
                    voting='soft',
                    weights=[1, 2, 1]
                )
            }

            if model_name not in model_config:
                raise ValueError(f"Invalid model: {model_name}")

            model = model_config[model_name]
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            metrics = {
                'accuracy': report['accuracy'] * 100,
                'precision': report['weighted avg']['precision'] * 100,
                'recall': report['weighted avg']['recall'] * 100,
                'f1': report['weighted avg']['f1-score'] * 100
            }

            os.makedirs("models", exist_ok=True)
            model_path = f"models/{model_name}.joblib"
            dump(model, model_path)
            dump(selector, "models/feature_selector.joblib")

            if not all(os.path.exists(f"models/{f}") for f in [f"{model_name}.joblib", "feature_selector.joblib"]):
                raise RuntimeError("Failed to save model artifacts")

            model_accuracies[model_name] = metrics
            return metrics

    except Exception as e:
        app.logger.error(f"Training failed for {model_name}: {str(e)}\n{traceback.format_exc()}")
        raise


def train_models():
    """Train all models (for initial setup)"""
    global model_accuracies
    model_accuracies = {}
    
    for model_name in ['SVM', 'RandomForest', 'NaiveBayes', 'Ensemble']:
        train_model(model_name)
    
    return model_accuracies

def generate_attack_distribution_plot():
    if not attack_history:
        return None
    
    attack_counts = pd.Series(attack_history).value_counts()
    plt.figure(figsize=(6, 4))
    plt.pie(attack_counts, labels=attack_counts.index, autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336', '#2196F3', '#FFC107', '#9C27B0', '#FF5722'])
    plt.title("Attack Type Distribution")
    
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def send_email(user_email, attack_type, detection_details):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG['sender_email']
    msg['To'] = user_email
    msg['Subject'] = "Cyberattack Detection Alert"
    
    body = f"""Dear User,
    
Our system has detected a {attack_type} attack on your network.
    
Detection Details:
{detection_details}
    
Please take the necessary security measures immediately.
    
Regards,
CyberSecurity Team"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.sendmail(EMAIL_CONFIG['sender_email'], user_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def packet_to_features(packet):
    """Convert raw packet to your selected features"""
    try:
        features = {
            'Duration': 0,  # Will calculate based on flow
            'Total_Fwd_Packets': 1,  # Will increment per flow
            'Total_Backward_Packets': 0,  # Will increment per flow
            'Down_Up_Ratio': 0,  # Will calculate
            'Average_Packet_Size': int(packet.length),
            'act_data_pkt_fwd': 1 if hasattr(packet, 'tcp') else 0,
            'min_seg_size_forward': int(getattr(packet.tcp, 'len', 0)) if hasattr(packet, 'tcp') else 0
        }
        return features
    except Exception as e:
        print(f"Error processing packet: {e}")
        return None

def capture_traffic(interface='eth0'):
    """Capture live network traffic"""
    global is_capturing
    try:
        capture = pyshark.LiveCapture(interface=interface)
        print(f"Starting capture on {interface}...")
        
        for packet in capture.sniff_continuously():
            if not is_capturing:
                break
                
            features = packet_to_features(packet)
            if features:
                traffic_queue.put(features)
                
    except Exception as e:
        print(f"Capture error: {e}")
    finally:
        print("Capture stopped")

def start_capture(interface='eth0'):
    """Start traffic capture in background thread"""
    global is_capturing
    if not is_capturing:
        is_capturing = True
        Thread(target=capture_traffic, args=(interface,), daemon=True).start()
        return True
    return False

def stop_capture():
    """Stop traffic capture"""
    global is_capturing
    is_capturing = False
    return True

def detect_attack(features_dict):
    """Detect attack from features using ensemble model"""
    try:
        # Convert to format your model expects
        features = [features_dict[feature] for feature in SELECTED_FEATURES]
        
        # Load models
        ensemble = load("models/Ensemble.joblib")
        selector = load("models/feature_selector.joblib")
        
        # Transform features
        features = selector.transform([features])
        
        # Make prediction
        prediction = ensemble.predict(features)[0]
        probabilities = ensemble.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        attack_type = ATTACK_TYPES.get(prediction, "Unknown")
        
        detection_result = {
            'attack_type': attack_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_used': 'Ensemble'
        }
        
        if attack_type != "Normal":
            attack_history.append(detection_result)  # This is the critical fix
            detection_details = f"""
            Detected {attack_type} with confidence {confidence:.2f}%
            Model used: Ensemble
            Parameters: {features_dict}
            """
            send_email(session.get('email', ''), attack_type, detection_details)
        
        return detection_result
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None
    

def calculate_feature_importances():
    """Calculate feature importances based on model weights"""
    try:
        # Load the ensemble model to get feature importances
        ensemble = load("models/Ensemble.joblib")
        
        # Get feature importances from the Random Forest model
        rf = ensemble.named_estimators_['rf']
        importances = rf.feature_importances_
        
        # Normalize to percentages
        importances = (importances / importances.max()) * 100
        
        # Create dictionary of feature importances
        return {
            feature: round(importance, 1)
            for feature, importance in zip(SELECTED_FEATURES, importances)
        }
    except Exception as e:
        print(f"Error calculating feature importances: {e}")
        # Fallback to equal distribution if there's an error
        return {
            feature: round(100/(i+2), 1)
            for i, feature in enumerate(SELECTED_FEATURES)
        }
    
def process_realtime_detections():
    """Process queued packets for real-time detection"""
    global realtime_detections
    while True:
        if not traffic_queue.empty():
            features_dict = traffic_queue.get()
            detection = detect_attack(features_dict)
            if detection:
                realtime_detections.append(detection)
                # Keep only the last MAX_REALTIME_DETECTIONS
                if len(realtime_detections) > MAX_REALTIME_DETECTIONS:
                    realtime_detections = realtime_detections[-MAX_REALTIME_DETECTIONS:]
        time.sleep(0.1)  # Prevent CPU overload

# Start the real-time processing thread
Thread(target=process_realtime_detections, daemon=True).start()

@app.context_processor
def inject_globals():
    return {
        'SELECTED_FEATURES': SELECTED_FEATURES,
        'ATTACK_TYPES': ATTACK_TYPES
    }

# ====================== Routes ======================
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('registration.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_registration WHERE Username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['username'] = username
            session['email'] = user[2]
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return render_template('register.html', error="Passwords don't match")
        
        hashed_password = generate_password_hash(password)
        
        conn = sqlite3.connect('registration.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO admin_registration (Username, Email, Password) VALUES (?, ?, ?)",
                          (username, email, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error="Username or email already exists")
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if not model_accuracies:
        train_models()
    
    dist_plot = generate_attack_distribution_plot()
    
    return render_template('dashboard.html',
                         username=session['username'],
                         model_accuracies=model_accuracies,
                         dist_plot=dist_plot,
                         attack_history=attack_history[-10:])
@app.route('/model_training')
def model_training():
    """Improved with session validation and error handling"""
    try:
        if 'username' not in session:
            return redirect(url_for('login'))
        
        return render_template('model_training.html',
                           username=session['username'],
                           model_accuracies=model_accuracies)
                           
    except Exception as e:
        app.logger.error(f"Model training page error: {str(e)}")
        return render_template('error.html', message="Failed to load training page"), 500

@app.route('/train_model/<model_name>', methods=['POST'])
def train_specific_model(model_name):
    """Training specific models using predefined dataset (no file upload)."""
    try:
        # Check user authentication
        if 'username' not in session:
            app.logger.warning("Unauthorized access attempt to train model")
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401

        # Validate model_name against supported models
        valid_models = ['SVM', 'RandomForest', 'NaiveBayes', 'Ensemble']
        if model_name not in valid_models:
            app.logger.warning(f"Invalid model requested: {model_name}")
            return jsonify({
                'success': False,
                'error': 'Invalid model',
                'valid_models': valid_models
            }), 400

        app.logger.info(f"Starting training for model: {model_name} using predefined dataset.")

        # Train using the predefined dataset
        metrics = train_model(model_name)

        app.logger.info(f"Training completed for model: {model_name} with metrics: {metrics}")

        return jsonify({
            'success': True,
            'model': model_name,
            'metrics': metrics,
            'updated_model_accuracies': model_accuracies,
            'message': f'{model_name} retrained successfully using predefined dataset'
        })

    except MemoryError:
        app.logger.error("MemoryError during training - possibly dataset too large")
        return jsonify({
            'success': False,
            'error': 'Out of memory',
            'message': 'Try with a smaller dataset'
        }), 500

    except Exception as e:
        app.logger.error(f"Training error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'See server logs for details'
        }), 500

@app.route('/detection')
def detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Calculate or get feature importances
    feature_importances = calculate_feature_importances()
    
    return render_template('detection.html',
                         username=session['username'],
                         SELECTED_FEATURES=SELECTED_FEATURES,
                         ATTACK_TYPES=ATTACK_TYPES,
                         FEATURE_IMPORTANCES=feature_importances)

@app.route('/detect_manual', methods=['POST'])
def detect_manual():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get random sample from the API
        api_response = requests.get('http://localhost:5000/api/get_random_sample')
        if not api_response.json().get('success'):
            raise ValueError("Failed to get sample from API")
        
        response_data = api_response.json()
        sample_data = response_data['sample']
        sample_index = response_data.get('sample_index')  # Get the index
        
        features = [sample_data[feature] for feature in SELECTED_FEATURES]
        
        # Use ensemble model for detection
        ensemble = load("models/Ensemble.joblib")
        selector = load("models/feature_selector.joblib")
        features = selector.transform([features])
        
        prediction = ensemble.predict(features)[0]
        probabilities = ensemble.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        attack_type = ATTACK_TYPES.get(prediction, "Unknown")
        actual_type = ATTACK_TYPES.get(sample_data['actual_label'], "Unknown")
        
        if attack_type != "Normal":
            attack_history.append(attack_type)
            detection_details = f"""
            Detected {attack_type} with confidence {confidence:.2f}%
            Model used: Ensemble
            Actual type was {actual_type}
            Sample index: {sample_index}
            Parameters used: {sample_data}
            """
            send_email(session['email'], attack_type, detection_details)
        
        dist_plot = generate_attack_distribution_plot()
        
        return jsonify({
            'attack_type': attack_type,
            'dist_plot': dist_plot,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_data': sample_data,
            'actual_type': actual_type,
            'correct': attack_type == actual_type,
            'model_used': 'Ensemble',
            'sample_index': sample_index  # Include the index in the response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/manual_detection')
def manual_detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('manual_detection.html',
                         username=session['username'],
                         SELECTED_FEATURES=SELECTED_FEATURES)

@app.route('/detect_from_input', methods=['POST'])
def detect_from_input():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get all features from form
        features_dict = {}
        for feature in SELECTED_FEATURES:
            value = request.form.get(feature)
            if not value:
                return jsonify({'error': f'Missing value for {feature}'}), 400
            features_dict[feature] = float(value)
        
        # Use ensemble model for detection
        ensemble = load("models/Ensemble.joblib")
        selector = load("models/feature_selector.joblib")
        
        # Prepare features in correct order
        features = [features_dict[feature] for feature in SELECTED_FEATURES]
        features = selector.transform([features])
        
        prediction = ensemble.predict(features)[0]
        probabilities = ensemble.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        attack_type = ATTACK_TYPES.get(prediction, "Unknown")
        
        if attack_type != "Normal":
            attack_history.append(attack_type)
            detection_details = f"""
            Detected {attack_type} with confidence {confidence:.2f}%
            Model used: Ensemble
            Parameters used: {features_dict}
            """
            send_email(session['email'], attack_type, detection_details)
        
        return jsonify({
            'attack_type': attack_type,
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_used': 'Ensemble',
            'features_used': features_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/realtime')
def realtime():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('realtime.html',
                         username=session['username'],
                         is_capturing=is_capturing)

@app.route('/start_realtime', methods=['POST'])
def start_realtime():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    interface = request.form.get('interface', 'eth0')
    if start_capture(interface):
        return jsonify({'status': 'started', 'interface': interface})
    return jsonify({'status': 'already running'})

@app.route('/stop_realtime', methods=['POST'])
def stop_realtime():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    stop_capture()
    return jsonify({'status': 'stopped'})

@app.route('/get_realtime_detections')
def get_realtime_detections():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    total_packets = len(realtime_detections)
    threats_detected = len([d for d in realtime_detections if d['attack_type'] != 'Normal'])

    return jsonify({
        'detections': realtime_detections[-20:],  # Return last 20 detections
        'is_capturing': is_capturing,
        'total_packets': total_packets,
        'threats_detected': threats_detected
    })

@app.route('/api/get_interfaces', methods=['GET'])
def get_interfaces():
    interfaces = list(psutil.net_if_addrs().keys())
    return jsonify({'interfaces': interfaces})

@app.route('/api/get_attack_types', methods=['GET'])
def get_attack_types():
    return jsonify({'attack_types': ATTACK_TYPES})



@app.route('/api/get_random_sample', methods=['GET'])
def get_random_sample():
    """API endpoint to get random sample from testing data"""
    try:
        # Load testing data (or use global test_data if you've already loaded it)
        test_data = pd.read_csv('test.csv')
        test_data = test_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Get random sample and its index
        random_sample = test_data.sample(1)
        sample_index = random_sample.index[0]  # Get the index of the selected row
        random_sample = random_sample.iloc[0]  # Get the row data
        
        # Convert the entire row to a dictionary
        sample_data = random_sample.to_dict()

        if 'Label' in sample_data and 'actual_label' not in sample_data:
            sample_data['actual_label'] = sample_data['Label']
        
        return jsonify({
            'success': True,
            'sample': sample_data,
            'sample_index': int(sample_index),  # Return the index for future reference
            'message': 'Random sample retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get random sample'
        }), 500

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('email', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    if not os.path.exists("models/Ensemble.joblib"):
        train_models()
    app.run(debug=True)

    