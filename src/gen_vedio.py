import requests
import uuid
import time
import os
import json
from datetime import datetime

def preprocess_audio(reference_audio_path):
    url = "http://127.0.0.1:18180/v1/preprocess_and_tran"
    
    
    payload = {
        "format": "wav",
        "reference_audio": reference_audio_path,
        "lang": "en"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  
        return response.json()
    except Exception as e:
        print(f"{str(e)}")
        return None

def synthesize_audio(text, preprocess_result, speaker_id):
    url = "http://127.0.0.1:18180/v1/invoke"
    
    payload = {
        "speaker": speaker_id,
        "text": text,
        "format": "wav",
        "topP": 0.7,
        "max_new_tokens": 1024,
        "chunk_length": 100,
        "repetition_penalty": 1.2,
        "temperature": 0.7,
        "need_asr": False,
        "streaming": False,
        "is_fixed_seed": 0,
        "is_norm": 0,
        "reference_audio": preprocess_result["asr_format_audio_url"],
        "reference_text": preprocess_result["reference_audio_text"]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        filename = save_synthesized_audio(response, speaker_id)
        return filename

    except Exception as e:
        print(f" {str(e)}")
        return None


def save_synthesized_audio(response, speaker_id, save_dir="/home/pc/heygem_data/face2face/temp/"):

    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"{speaker_id}_{timestamp}.wav"
    save_path = os.path.join(save_dir, filename)
    
    with open(save_path, "wb") as f:
        f.write(response.content)
    
    return filename


def synthesize_video(audio_url, video_url, speaker_id):
    url = "http://127.0.0.1:8383/easy/submit"
    
    payload = {
        "audio_url": audio_url,
        "video_url": video_url,
        "code": speaker_id, 
        "chaofen": 0,
        "watermark_switch": 0,
        "pn": 1
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return speaker_id  
    except Exception as e:
        print(f"{str(e)}")
        return None

def check_video_progress(task_code):
    url = f"http://127.0.0.1:8383/easy/query?code={task_code}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f" {str(e)}")
        return None

def main():
    ref_video_path = "ref_face.mp4"
    
    
    text_file = "/home/pc/lyy/ICH-agent/vedio_data/patient_results.json"
    with open(text_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, d in enumerate(data):
        audio_result = "{:03d}".format(i+1) + ".wav"
        task_code = synthesize_video(audio_result, ref_video_path, "{:03d}".format(i+1))
        if not task_code:
            return

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    run_time = end_time - start_time

