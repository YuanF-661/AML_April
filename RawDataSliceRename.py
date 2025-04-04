import os
import librosa
import soundfile as sf
import numpy as np
from collections import defaultdict

def auto_trim_all_recursive(input_root, output_root, sr=44100, head_padding=0.02, tail_padding=0.1, threshold_db=-40):
    os.makedirs(output_root, exist_ok=True)
    index_counters = defaultdict(int)  # 每种 category_machine 一个 index

    for root, _, files in os.walk(input_root):
        for file in files:
            if not file.lower().endswith((".wav", ".mp3", ".flac", ".aiff", ".aif")):
                continue

            input_path = os.path.join(root, file)
            filename_wo_ext = os.path.splitext(file)[0]
            name_parts = filename_wo_ext.split("_")

            if len(name_parts) < 2:
                print(f"跳过无法识别的文件名: {file}")
                continue

            category = name_parts[0]
            machine = name_parts[1]
            folder_path = os.path.join(output_root, machine, category)
            os.makedirs(folder_path, exist_ok=True)

            # 读取音频
            try:
                y, _ = librosa.load(input_path, sr=sr)
            except Exception as e:
                print(f"加载失败 {file}: {e}")
                continue

            # 获取非静音区间
            intervals = librosa.effects.split(y, top_db=abs(threshold_db))
            if len(intervals) == 0:
                print(f"跳过静音文件: {file}")
                continue

            start_sample = max(0, intervals[0][0] - int(head_padding * sr))
            end_sample = min(len(y), intervals[-1][1] + int(tail_padding * sr))
            trimmed = y[start_sample:end_sample]

            # 每种类别单独编号
            key = f"{category}_{machine}"
            index_counters[key] += 1
            current_index = index_counters[key]

            new_filename = f"{key}_Raw_{current_index:03d}.wav"
            output_path = os.path.join(folder_path, new_filename)

            sf.write(output_path, trimmed, sr)
            print(f"保存: {output_path}")


# 用法
input_folder = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/Roland_TR_808_Full/808_Raw_Full"     # 替换为你的输入文件夹
output_folder = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/808_Raw_Sliced" # 替换为输出文件夹
auto_trim_all_recursive(input_folder, output_folder)

