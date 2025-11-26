import static_ffmpeg
static_ffmpeg.add_paths() 

from flask import Flask, request, jsonify
from flask_cors import CORS   
import os 
import librosa
import numpy as np
import requests
import psutil
from fastdtw import fastdtw
from scipy.spatial.distance import cosine 
import math # Добавил для формулы

app = Flask(__name__)
CORS(app) 

def log_memory_usage(stage=""):
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f"--- MEMORY USAGE [{stage}]: {memory_mb:.2f} MB", flush=True)
    except:
        pass

def compare_pronunciation(original_file_path, user_file_path):
    try:
        TARGET_SR = 16000 
        
        y_orig, _ = librosa.load(original_file_path, sr=TARGET_SR, mono=True)
        y_user, _ = librosa.load(user_file_path, sr=TARGET_SR, mono=True)
        
        # Обрезка тишины (оставляем top_db=30, это хорошо работает)
        y_orig, _ = librosa.effects.trim(y_orig, top_db=30)
        y_user, _ = librosa.effects.trim(y_user, top_db=30)

        if len(y_user) < 1000:
             return {"similarity": 0, "status": "success", "message": "Too silent"}

        # MFCC
        mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=TARGET_SR, n_mfcc=13, hop_length=512)
        mfcc_user = librosa.feature.mfcc(y=y_user, sr=TARGET_SR, n_mfcc=13, hop_length=512)
        
        # CMVN (Нормализация)
        mfcc_orig = (mfcc_orig - np.mean(mfcc_orig, axis=1, keepdims=True)) / (np.std(mfcc_orig, axis=1, keepdims=True) + 1e-8)
        mfcc_user = (mfcc_user - np.mean(mfcc_user, axis=1, keepdims=True)) / (np.std(mfcc_user, axis=1, keepdims=True) + 1e-8)

        # DTW Cosine
        distance, path = fastdtw(mfcc_orig.T, mfcc_user.T, dist=cosine, radius=50)
        
        path_len = len(path) if len(path) > 0 else 1
        normalized_distance = distance / path_len
        
        print(f"=== COSINE DISTANCE: {normalized_distance:.4f} ===", flush=True)
        
        # --- НОВАЯ "ДОБРАЯ" КАЛИБРОВКА (Sigmoid) ---
        # Центр кривой смещен в 0.55. 
        # Всё что до 0.45 - считается хорошим.
        # Всё что после 0.65 - считается плохим.
        
        # 0.20 -> 97%
        # 0.42 -> 79% (Твой текущий результат)
        # 0.55 -> 50%
        # 0.80 -> 7%
        
        similarity = 100 / (1 + math.exp(10 * (normalized_distance - 0.55)))
        
        # Небольшой бонус за попытку, если не совсем ужасно
        if normalized_distance < 0.6 and similarity < 40:
            similarity = 40

        return {
            "similarity": round(similarity),
            "status": "success"
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return {
            "status": "error",
            "message": str(e)
        }

@app.route('/compare-audio', methods=['POST'])
def compare_audio_files():
    if 'user_audio' not in request.files: return jsonify({"status": "error"}), 400
    if 'original_video_url' not in request.form: return jsonify({"status": "error"}), 400

    user_file = request.files['user_audio']
    original_video_url = request.form['original_video_url']

    upload_folder = 'temp_uploads'
    if not os.path.exists(upload_folder): os.makedirs(upload_folder)
        
    user_path = os.path.join(upload_folder, "user_temp.webm")
    original_path = os.path.join(upload_folder, "original_temp_video") 

    try:
        user_file.save(user_path)
        r = requests.get(original_video_url, stream=True)
        r.raise_for_status()
        with open(original_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        result = compare_pronunciation(original_path, user_path)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(user_path): os.remove(user_path)
        if os.path.exists(original_path): os.remove(original_path)
            
    return jsonify(result)

@app.route('/')
def home():
    return "Audio Server (Sigmoid Calibration)"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)