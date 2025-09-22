from TTS.api import TTS
import json
from datetime import datetime
import re
import time


def convert_medical_units(text):

    text = re.sub(r'(\d+)\s*(mL|ml)', r'\1 milliliters', text)
    

    def replace_blood_pressure(match):
        systolic = match.group(1)
        diastolic = match.group(2)
        return f"a systolic blood pressure of {systolic} millimeters of mercury and a diastolic blood pressure of {diastolic} millimeters of mercury"
    
    text = re.sub(r'(\d+)/(\d+)\s*mmHg', replace_blood_pressure, text)
    
    text = re.sub(r'(\d+)\s*(mmHg)', r'\1 millimeters of mercury', text)

    text = re.sub(r'(\d+)\s*μg/L', r'\1 micrograms per liter', text)
    
    return text

def clean_text(text):

    cleaned = text.replace('*', '').replace('-', '').replace('#', '')

    cleaned = cleaned.replace('Dear [Patient’s Name],', '')
    cleaned = cleaned.replace('Dear [Patient\'s Name],', '')
    
    cleaned = cleaned.replace('AHA/ASA', '')

    cleaned = cleaned.replace('AHA', '')

    cleaned = re.sub(r'\([^)]*\)', '', cleaned)

    cleaned = convert_medical_units(cleaned)

    cleaned = cleaned.replace('ICH', 'intracerebral hemorrahge')

    cleaned = re.sub(r' +', ' ', cleaned).strip()
    cleaned = re.sub(r'\n+', '\n', cleaned)

    return cleaned


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
text_file = "/home/pc/lyy/ICH-agent/vedio_data/patient_results.json"

with open(text_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


start_time = time.time()
for i, d in enumerate(data):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{i+1}:", formatted_time)

    d['content'] = clean_text(d['content'])

    tts.tts_to_file(text=d['content'],
                file_path="/home/pc/lyy/ICH-agent/agent/gen_audio_data_zxj/{:03d}.wav".format(i+1),
                speaker_wav="/home/pc/lyy/ICH-agent/zxj.wav",
                language="en")
    
end_time = time.time()
run_time = end_time - start_time
print(f"{run_time:.6f} 秒")


