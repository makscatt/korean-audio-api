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
from scipy.spatial.distance import euclidean
import warnings

# Игнорируем варнинги, чтобы не засорять логи
warnings.filterwarnings("ignore")

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
        
        # 2. Удаление тишины (очень важно)
        y_orig, _ = librosa.effects.trim(y_orig, top_db=20)
        y_user, _ = librosa.effects.trim(y_user, top_db=20)

        # 3. Нормализация ГРОМКОСТИ (чтобы оба были максимально громкими)
        y_orig = librosa.util.normalize(y_orig)
        y_user = librosa.util.normalize(y_user)

        if len(y_user) < 1000:
             return {"similarity": 0, "status": "success", "message": "Too silent"}

        # 4. Извлечение MFCC
        # n_mfcc=13 - стандарт для речи
        mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=TARGET_SR, n_mfcc=13)
        mfcc_user = librosa.feature.mfcc(y=y_user, sr=TARGET_SR, n_mfcc=13)
        
        # 5. CMS (Cepstral Mean Subtraction) - САМОЕ ВАЖНОЕ
        # Вычитаем среднее значение, чтобы убрать влияние микрофона/канала
        mfcc_orig = mfcc_orig - np.mean(mfcc_orig, axis=1, keepdims=True)
        mfcc_user = mfcc_user - np.mean(mfcc_user, axis=1, keepdims=True)

        # 6. DTW
        distance, path = fastdtw(mfcc_orig.T, mfcc_user.T, dist=euclidean)
        
        # 7. Нормализация дистанции
        path_len = len(path) if len(path) > 0 else 1
        normalized_distance = distance / path_len
        
        print(f"=== DISTANCE (After CMS): {normalized_distance:.4f} ===", flush=True)
        
        # 8. Новая формула оценки
        # После CMS дистанции станут меньше (примерно 20-50)
        # 15-25 -> Отлично (90-100%)
        # 35 -> Хорошо (75%)
        # 50 -> Средне (50%)
        # >60 -> Плохо
        
        # Используем экспоненту для плавного спуска
        similarity = 100 * np.exp(-(normalized_distance - 18) / 40)
        
        # Обрезаем границы
        if similarity > 100: similarity = 100
        if similarity < 0: similarity = 0
        if normalized_distance < 22: similarity = 95 + np.random.randint(0, 5) # Бонус за супер-точность

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
    log_memory_usage("Start")
    
    if 'user_audio' not in request.files:
        return jsonify({"status": "error", "message": "No user_audio"}), 400
    if 'original_video_url' not in request.form:
        return jsonify({"status": "error", "message": "No original_video_url"}), 400

    user_file = request.files['user_audio']
    original_video_url = request.form['original_video_url']

    upload_folder = 'temp_uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
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
    return "Audio Server is Live (Updated CMS)"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)