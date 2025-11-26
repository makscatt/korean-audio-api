import static_ffmpeg
static_ffmpeg.add_paths() # Подключаем кодеки

from flask import Flask, request, jsonify
from flask_cors import CORS   
import os 
import librosa
import numpy as np
import requests
import psutil
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

app = Flask(__name__)
CORS(app) 

def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"--- MEMORY USAGE [{stage}]: {memory_mb:.2f} MB", flush=True)

def compare_pronunciation(original_file_path, user_file_path):
    try:
        # Настройки для Librosa (баланс качества и скорости)
        TARGET_SR = 16000 
        
        # 1. Загрузка аудио через Librosa (она сама извлечет звук из видео)
        # mono=True смешивает каналы, duration=15 ограничивает длину (защита памяти)
        original_audio, _ = librosa.load(original_file_path, sr=TARGET_SR, mono=True, duration=15)
        user_audio, _ = librosa.load(user_file_path, sr=TARGET_SR, mono=True, duration=15)
        
        # 2. Удаление тишины (librosa делает это очень качественно)
        original_audio, _ = librosa.effects.trim(original_audio, top_db=20)
        user_audio, _ = librosa.effects.trim(user_audio, top_db=20)

        # Если файл пустой после обрезки
        if len(user_audio) < 1000:
             return {"similarity": 0, "status": "success", "message": "Too silent"}

        # 3. Извлечение MFCC (13 коэффициентов достаточно для речи)
        original_mfcc = librosa.feature.mfcc(y=original_audio, sr=TARGET_SR, n_mfcc=13)
        user_mfcc = librosa.feature.mfcc(y=user_audio, sr=TARGET_SR, n_mfcc=13)
        
        # 4. Сравнение через FastDTW
        # Транспонируем (.T), так как fastdtw ждет (N, 13)
        distance, path = fastdtw(original_mfcc.T, user_mfcc.T, dist=euclidean)
        
        # 5. Расчет процента
        # Нормализация дистанции
        path_len = len(path) if len(path) > 0 else 1
        normalized_distance = distance / path_len
        
        print(f"=== RAW DISTANCE: {normalized_distance} ===", flush=True)
        
        # Формула для Librosa MFCC + DTW (обычно дистанция около 20-50)
        # 20 -> 100%
        # 35 -> 70%
        # 50 -> 40%
        similarity = 100 * np.exp(-(normalized_distance - 20) / 30)
        if similarity > 100: similarity = 100
        if similarity < 0: similarity = 0

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
    log_memory_usage("Start Request")
    
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
    original_path = os.path.join(upload_folder, "original_temp_video") # Librosa сама поймет формат

    try:
        user_file.save(user_path)
        
        # Скачиваем оригинал
        r = requests.get(original_video_url, stream=True)
        r.raise_for_status()
        with open(original_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        log_memory_usage("Files Saved")
        
        # Сравниваем
        result = compare_pronunciation(original_path, user_path)
        log_memory_usage("Done")

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(user_path): os.remove(user_path)
        if os.path.exists(original_path): os.remove(original_path)
            
    return jsonify(result)

@app.route('/')
def home():
    return "Audio Server is Running (Librosa)"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)