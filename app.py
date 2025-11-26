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
        
        # 1. Загрузка
        y_orig, _ = librosa.load(original_file_path, sr=TARGET_SR, mono=True)
        y_user, _ = librosa.load(user_file_path, sr=TARGET_SR, mono=True)
        
        # 2. УМНАЯ ОБРЕЗКА (Stricter Trimming)
        # top_db=30 означает, что всё, что тише основного голоса на 30дБ, считается тишиной.
        # Это обрежет шум вентилятора или дыхание до/после слов.
        y_orig, _ = librosa.effects.trim(y_orig, top_db=30)
        y_user, _ = librosa.effects.trim(y_user, top_db=30)

        # Если после обрезки ничего не осталось
        if len(y_user) < 1000:
             return {"similarity": 0, "status": "success", "message": "Too silent"}

        # 3. Извлечение MFCC
        # Используем hop_length=512 для детальности
        mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=TARGET_SR, n_mfcc=13, hop_length=512)
        mfcc_user = librosa.feature.mfcc(y=y_user, sr=TARGET_SR, n_mfcc=13, hop_length=512)
        
        # 4. CMVN (Cepstral Mean and Variance Normalization)
        # Приводим к стандарту (среднее 0, разброс 1). 
        # Это делает сравнение независимым от громкости и микрофона.
        mfcc_orig = (mfcc_orig - np.mean(mfcc_orig, axis=1, keepdims=True)) / (np.std(mfcc_orig, axis=1, keepdims=True) + 1e-8)
        mfcc_user = (mfcc_user - np.mean(mfcc_user, axis=1, keepdims=True)) / (np.std(mfcc_user, axis=1, keepdims=True) + 1e-8)

        # 5. DTW с Косинусной метрикой
        # radius=50 дает алгоритму свободу сопоставить медленную речь с быстрой
        distance, path = fastdtw(mfcc_orig.T, mfcc_user.T, dist=cosine, radius=50)
        
        # 6. Нормализация
        path_len = len(path) if len(path) > 0 else 1
        normalized_distance = distance / path_len
        
        print(f"=== COSINE DISTANCE: {normalized_distance:.4f} ===", flush=True)
        
        # 7. Расчет оценки (Калибровка под Cosine + CMVN)
        # 0.15 - 0.25 -> Отлично (90-100%)
        # 0.25 - 0.40 -> Хорошо (70-90%)
        # 0.40 - 0.60 -> Средне (40-70%)
        # > 0.60 -> Плохо
        
        # Формула Гаусса (колокол) с центром в 0
        # Чем дальше дистанция от 0, тем быстрее падает оценка
        similarity = 100 * np.exp(-(normalized_distance**2) / (2 * (0.35**2)))
        
        # Немного подтянем оценки, если они слишком строгие
        if normalized_distance < 0.4:
            similarity += 10 # Бонус за старание
            
        if similarity > 100: similarity = 100
        
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
    # Без изменений
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
    return "Audio Server (Auto-Trim + Cosine)"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)