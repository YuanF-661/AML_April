import os
import random
import numpy as np
import librosa
import soundfile as sf
from glob import glob

def get_augmented_filename(original_file, global_number):
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    if '_' in base_name:
        parts = base_name.rsplit('_', 1)
        category = parts[0]
        orig_index = parts[1]
    else:
        category = base_name
        orig_index = "00"
    try:
        orig_index_int = int(orig_index)
    except ValueError:
        orig_index_int = 0
    new_filename = f"{category}_{global_number:02d}_aug{orig_index_int:02d}.wav"
    return new_filename

def time_stretch_audio(y, rate):
    D = librosa.stft(y)
    D_stretch = librosa.phase_vocoder(D, rate=rate)
    return librosa.istft(D_stretch)

def spectral_perturbation_audio(y, sr, noise_factor=0.008):
    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)
    noise = noise_factor * np.random.randn(*magnitude.shape)
    magnitude_noisy = magnitude + noise
    D_noisy = magnitude_noisy * np.exp(1j * phase)
    return librosa.istft(D_noisy)

def augment_audio_instance(y, sr):
    method = random.choice(['time_stretch', 'spectral_perturb'])
    if method == 'time_stretch':
        rate = random.uniform(0.8, 1.2)
        return time_stretch_audio(y, rate)
    else:
        noise_factor = random.uniform(0.001, 0.01)
        return spectral_perturbation_audio(y, sr, noise_factor)

def parse_category_and_drum(file_name):
    parts = file_name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "Unknown", "Unknown"

def balance_and_augment(input_dir, output_dir, target_count=30):
    audio_files = sorted(glob(os.path.join(input_dir, '**', '*.wav'), recursive=True))
    print(f"总共发现音频文件：{len(audio_files)}")

    # 按 (drummachine, category) 分类
    category_dict = {}
    for file in audio_files:
        base = os.path.basename(file)
        cat, drum = parse_category_and_drum(base)
        key = (drum, cat)
        category_dict.setdefault(key, []).append(file)

    for (drum, cat), files in category_dict.items():
        output_subdir = os.path.join(output_dir, drum, cat)
        os.makedirs(output_subdir, exist_ok=True)

        print(f"处理: {drum}/{cat}，已有样本: {len(files)}")

        # 保存原始文件
        for file in files:
            y, sr = librosa.load(file, sr=None)
            name = os.path.basename(file)
            out_path = os.path.join(output_subdir, name)
            sf.write(out_path, y, sr, subtype='PCM_16')

        # 增强补齐
        current_count = len(files)
        aug_index = 1
        while current_count < target_count:
            file = random.choice(files)
            y, sr = librosa.load(file, sr=None)
            y_aug = augment_audio_instance(y, sr)
            global_number = current_count + 1
            new_filename = get_augmented_filename(file, global_number)
            out_path = os.path.join(output_subdir, new_filename)
            sf.write(out_path, y_aug, sr, subtype='PCM_16')
            print(f"[+] 增强并保存: {new_filename}")
            current_count += 1
            aug_index += 1

        print(f"[✓] 处理完成: {drum}/{cat}（共 {current_count} 个样本）")

if __name__ == "__main__":
    input_dir = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/Linn_RawData/Perc"         # 修改为你的输入文件夹路径
    output_dir = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/Linn_RawDataNew/Perc"  # 修改为你的输出路径
    target_count = 150               # 每个类别希望达到的最小样本数量

    balance_and_augment(input_dir, output_dir, target_count)
