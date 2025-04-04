import os
import librosa
import soundfile as sf
from collections import defaultdict

def autoslice_and_rename(input_folder, output_folder, file_ext=".wav",
                         sr=44100, head_padding=0.02, tail_padding=0.1, threshold_db=-40):
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 收集所有文件（递归）
    all_files = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.endswith(file_ext):
                full_path = os.path.join(root, f)
                all_files.append(full_path)

    all_files.sort()

    # 用于记录每个 chain+category+machine 的编号索引
    index_counters = defaultdict(int)

    for file_path in all_files:
        filename = os.path.basename(file_path)
        name_part = os.path.splitext(filename)[0]
        parts = name_part.split("-")

        if len(parts) != 2:
            print(f"文件名格式错误：{filename}")
            continue

        left, right = parts
        left = left.strip()

        try:
            chain = left.split("_")[1]  # 提取 chain，如 CC1
        except:
            print(f"无法解析处理链：{filename}")
            continue

        cat_parts = right.split("_")
        if len(cat_parts) < 2:
            print(f"无法解析鼓类别和鼓机类型：{filename}")
            continue

        category = cat_parts[0]    # Cymbal
        machine = cat_parts[1]     # 808

        # 输出文件夹结构
        subfolder = os.path.join(output_folder, machine, category)
        os.makedirs(subfolder, exist_ok=True)

        # autoslice：加载音频并裁剪
        try:
            y, _ = librosa.load(file_path, sr=sr)
        except Exception as e:
            print(f"加载失败 {filename}: {e}")
            continue

        intervals = librosa.effects.split(y, top_db=abs(threshold_db))
        if len(intervals) == 0:
            print(f"跳过静音文件: {filename}")
            continue

        start_sample = max(0, intervals[0][0] - int(head_padding * sr))
        end_sample = min(len(y), intervals[-1][1] + int(tail_padding * sr))
        trimmed_audio = y[start_sample:end_sample]

        # 命名计数器 key
        key = f"{chain}_{category}_{machine}"
        index_counters[key] += 1
        suffix = f"{index_counters[key]:03d}"

        new_filename = f"{category}_{machine}_Ana_{chain}_{suffix}{file_ext}"
        output_path = os.path.join(subfolder, new_filename)

        # 保存裁剪后的音频
        sf.write(output_path, trimmed_audio, sr)
        print(f"✅ {filename} -> {output_path}")

# 使用方法
input_folder = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/Roland_TR_808_Full/808_Processed_Full"     # 修改为你的输入路径
output_folder = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/808_Processed_Sliced"   # 修改为你的输出路径
autoslice_and_rename(input_folder, output_folder)

# test 0404 20:47