import os
import numpy as np
import librosa
import pickle
from flask import Flask, request, render_template, redirect, url_for, session,make_response
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
from datetime import timedelta, datetime, timezone #time for token,expire of session
import random
import string
import time

from pydub import AudioSegment  # Import pydub for audio conversion
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Define Flask app
app = Flask(__name__)
app.secret_key = '624512'  # Change this to a random secret key

# Path for file uploads
UPLOAD_FOLDER = 'D://data//tested audioss'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac',"aac",'m4a','webm'}  # Add other allowed formats

# Load model, label encoder, and scaler
model = load_model('emotion_stream.h5')
with open('labels.pkl', 'rb') as infile:
    lb = pickle.load(infile)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_noise(signal, noise_factor=0.005, random=False):
    if random:
        noise = np.random.randn(len(signal))
        return signal + noise_factor * noise
    return signal

def pitching(signal, sr, pitch_factor=0.5, random=False):
    if random:
        n_steps = np.random.uniform(-pitch_factor, pitch_factor)
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
    return signal

def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result

def get_features(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    audio = extract_features(data, sr)
    audio = np.array(audio)

    # Extract features for augmented data
    audio_list = [audio]

    # Add noise
    noised_audio = add_noise(data, random=True)
    aud2 = extract_features(noised_audio, sr)
    audio_list.append(aud2)

    # Pitch modification
    pitched_audio = pitching(data, sr, random=True)
    aud3 = extract_features(pitched_audio, sr)
    audio_list.append(aud3)

    # Pitch modification with noise
    pitched_audio1 = pitching(data, sr, random=True)
    pitched_noised_audio = add_noise(pitched_audio1, random=True)
    aud4 = extract_features(pitched_noised_audio, sr)
    audio_list.append(aud4)

    # Stack all features and ensure correct shape
    return np.vstack(audio_list)

def convert_to_wav(file_path, output_path):
    """Convert audio file to WAV format using pydub"""
    audio = AudioSegment.from_file(file_path)
    audio.export(output_path, format="wav")
    return output_path

# Routes
@app.route('/')
def index():
    return render_template('new.html')
# Spotify API Credentials
CLIENT_ID = '428bbcecfbdd4e22a0e1eec5adeed462'
CLIENT_SECRET = 'c03768adfc544079894dedaf4dcb8ca8'
REDIRECT_URI = 'http://127.0.0.1:5000/callback'

# Refresh access token
def refresh_token(refresh_token):
    token_url = 'https://accounts.spotify.com/api/token'
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        token_info = response.json()
        session['access_token'] = token_info.get('access_token')
        session['refresh_token'] = token_info.get('refresh_token', session.get('refresh_token'))
       #session['expires_at'] = (datetime.now(datetime.timezone.utc) + timedelta(seconds=token_info.get('expires_in'))).timestamp()
        session['expires_at'] = (datetime.now(timezone.utc) + timedelta(seconds=token_info.get('expires_in'))).timestamp()

        return True
    return False

@app.route('/authorize')
def authorize():
    # Only authorize if no access token or expired token exists
    if 'access_token' in session and session.get('expires_at', 0) > time.time():
        return redirect(url_for('index'))
    
    # If there's a refresh token, attempt to refresh it
    if 'refresh_token' in session:
        if refresh_token(session['refresh_token']):
            return redirect(url_for('index'))

    # Generate a new authorization flow if no valid tokens exist
    state = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    session['oauth_state'] = state
    scopes = 'user-read-private user-read-email streaming user-read-playback-state user-modify-playback-state'
    auth_url = f'https://accounts.spotify.com/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope={scopes}&state={state}'
    return redirect(auth_url)

@app.route('/callback')
def callback():
    if request.args.get('state') != session.get('oauth_state'):
        return "Invalid state parameter", 400
    code = request.args.get('code')
    token_url = 'https://accounts.spotify.com/api/token'
    payload = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(token_url, data=payload)
    if response.status_code != 200:
        return f"Error fetching access token: {response.json()}", 500

    token_info = response.json()
    session['access_token'] = token_info.get('access_token')
    session['refresh_token'] = token_info.get('refresh_token')
    #session['expires_at'] = (datetime.now(datetime.timezone.utc) + timedelta(seconds=token_info.get('expires_in'))).timestamp()
    session['expires_at'] = (datetime.now(timezone.utc) + timedelta(seconds=token_info.get('expires_in'))).timestamp()

    response = make_response(redirect(url_for('index')))
    response.set_cookie('login', 'true', max_age=3600)
    return response

@app.route('/record', methods=['POST'])
def record():
    filename = request.args.get('filename')
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # If user does not select file, browser may also submit an empty part without filename
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check if the uploaded file is not a WAV file and convert it
        if not file.filename.lower().endswith('.wav'):
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
            file_path = convert_to_wav(file_path, wav_path)  # Convert to WAV

        # Extract features
        X_real_data = get_features(file_path)

        # Ensure the number of features matches 2376 (by padding if needed)
        if X_real_data.shape[1] < 2376:
            X_real_data = np.pad(X_real_data, ((0, 0), (0, 2376 - X_real_data.shape[1])), mode='constant')
        elif X_real_data.shape[1] > 2376:
            X_real_data = X_real_data[:, :2376]  # Truncate if too many features

        # Scale the features using the previously fitted scaler
        X_real_data_scaled = scaler.transform(X_real_data)

        # Reshape the data for model input
        X_real_data_scaled = X_real_data_scaled.reshape(-1, 2376, 1)

        # Make predictions
        y_pred_real = model.predict(X_real_data_scaled)

        # Convert predictions to class indices
        predicted_indices = np.argmax(y_pred_real, axis=1)

        # Get the most common predicted class (emotion)
       # final_prediction_index = mode(predicted_indices).mode[0]
        final_prediction_index = np.bincount(predicted_indices).argmax()
        final_prediction_emotion = lb.inverse_transform([final_prediction_index])
        
        # Return the result
        access_token = session.get('access_token')
    
        return redirect(url_for('check_premium', emotion=final_prediction_emotion[0], access_token=access_token))





@app.route('/predict', methods=['POST'])
def predict():
    filename = request.args.get('filename')
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # If user does not select file, browser may also submit an empty part without filename
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check if the uploaded file is not a WAV file and convert it
        if not file.filename.lower().endswith('.wav'):
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
            file_path = convert_to_wav(file_path, wav_path)  # Convert to WAV

        # Extract features
        X_real_data = get_features(file_path)

        # Ensure the number of features matches 2376 (by padding if needed)
        if X_real_data.shape[1] < 2376:
            X_real_data = np.pad(X_real_data, ((0, 0), (0, 2376 - X_real_data.shape[1])), mode='constant')
        elif X_real_data.shape[1] > 2376:
            X_real_data = X_real_data[:, :2376]  # Truncate if too many features

        # Scale the features using the previously fitted scaler
        X_real_data_scaled = scaler.transform(X_real_data)

        # Reshape the data for model input
        X_real_data_scaled = X_real_data_scaled.reshape(-1, 2376, 1)

        # Make predictions
        y_pred_real = model.predict(X_real_data_scaled)

        # Convert predictions to class indices
        predicted_indices = np.argmax(y_pred_real, axis=1)

        # Get the most common predicted class (emotion)
       # final_prediction_index = mode(predicted_indices).mode[0]
        final_prediction_index = np.bincount(predicted_indices).argmax()
        final_prediction_emotion = lb.inverse_transform([final_prediction_index])
        
        # Return the result
        access_token = session.get('access_token')
    
        return redirect(url_for('check_premium', emotion=final_prediction_emotion[0], access_token=access_token))


# Check Spotify premium status and redirect accordingly
@app.route('/check-premium', methods=['GET'])
def check_premium():
    access_token = session.get('access_token')
    detected_emotion = request.args.get('emotion')
    
    if not access_token:
        return "Error: Access token is missing", 400

    profile = get_spotify_profile(access_token)
    if profile is None:
        return "Error: Unable to fetch user profile", 500

    if profile.get('product') == 'premium':
        return redirect(url_for('result', access_token=access_token, emotion=detected_emotion))
    else:
        return redirect(url_for('normal', access_token=access_token, emotion=detected_emotion))

# Get Spotify user profile
def get_spotify_profile(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get("https://api.spotify.com/v1/me", headers=headers)
    return response.json() if response.status_code == 200 else None

# Normal user page
@app.route('/normal', methods=['GET'])
def normal():
    emotion = request.args.get('emotion')
    access_token = request.args.get('access_token')
    return render_template('normal.html', emotion=emotion, access_token=access_token)

# Premium user page
@app.route('/result')
def result():
    emotion = request.args.get('emotion')
    access_token = session.get('access_token')
    return render_template('result.html', emotion=emotion, access_token=access_token)

# Handle user logout and optionally revoke token
@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    response = make_response(redirect(url_for('index')))
    response.set_cookie('login', '', expires=0)  # Clear the login cookie
    return response

if __name__ == "__main__":
    # Ensure the upload directory exists
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True) 
if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')