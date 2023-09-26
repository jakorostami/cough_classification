import os
import subprocess
from multiprocessing import Pool, cpu_count

kaggle_path = "C:/Users/jako/data/kaggle_cough"  # Replace with your path
output_path = "C:/Users/jako/data/kaggle_cough/wav_format"  # Replace with your desired output path

MAX_SIZE = 100 * 1024


def convert_webm_to_wav(webm):
    if webm.endswith(".webm"):
        WEBM_FILE = os.path.join(kaggle_path, webm)
        
        # Check the file size
        if os.path.getsize(WEBM_FILE) > MAX_SIZE:
            return f"Skipping {webm} because it's larger than 100 kB."
        
        output_name = os.path.join(output_path, webm[:-5] + ".wav")
        
        command = [
            "ffmpeg", "-i", WEBM_FILE,
            "-vn",
            "-acodec", "pcm_s16le",  # 16-bit depth
            "-ac", "1",  # Mono
            "-ar", "16000",  # 44.1 kHz sample rate
            "-f", "wav", 
            output_name
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return f"Successfully converted {webm} to {output_name}"
        except subprocess.CalledProcessError as e:
            return f"Error processing {webm}: {e.stderr.decode('utf-8')}"
    return None

if __name__ == '__main__':
    webm_files = [f for f in os.listdir(kaggle_path) if f.endswith(".webm")]

    with Pool(processes=cpu_count() // 2) as pool:  # Using half the CPU cores
        results = pool.map(convert_webm_to_wav, webm_files)
        for result in results:
            if result:
                print(result)