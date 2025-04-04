import os
import random
import numpy as np
import soundfile as sf
from glob import glob
from pedalboard import Pedalboard, HighShelfFilter, LowShelfFilter, Compressor, Reverb, Gain, Phaser

def get_chain_filename(original_file, global_number, chain_number):
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
    new_filename = f"{category}_{global_number:02d}_chain{chain_number}.wav"
    return new_filename

def parse_category_and_drum(file_name):
    parts = file_name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "Unknown", "Unknown"

def apply_pedalboard_chain_1(audio, sr):
    board = Pedalboard([
        Gain(gain_db=3.0),
        LowShelfFilter(cutoff_frequency_hz=100, gain_db=6.0, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=4000, gain_db=4.0, q=0.7),
        Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=5.0, release_ms=50.0),
        Reverb(room_size=0.6, damping=0.3, wet_level=0.25),
    ])
    return board(audio, sr)

def apply_pedalboard_chain_2(audio, sr):
    board = Pedalboard([
        Gain(gain_db=-2.0),
        Phaser(rate_hz=0.3),
        Reverb(room_size=0.5, damping=0.5, wet_level=0.3),
    ])
    return board(audio, sr)

def apply_pedalboard_chain_3(audio, sr):
    board = Pedalboard([
        Gain(gain_db=6.0),
        HighShelfFilter(cutoff_frequency_hz=5000, gain_db=8.0, q=0.7),
        Compressor(threshold_db=-12.0, ratio=4.0, attack_ms=10.0, release_ms=100.0),
    ])
    return board(audio, sr)

def apply_pedalboard_chain_4(audio, sr):
    board = Pedalboard([
        Gain(gain_db=-3.0),
        LowShelfFilter(cutoff_frequency_hz=80, gain_db=5.0, q=0.7),
        Phaser(rate_hz=0.2),
        Reverb(room_size=0.8, damping=0.4, wet_level=0.4),
    ])
    return board(audio, sr)

def process_all_with_pedalboard(input_dir, output_dir, chosen_chains):
    audio_files = sorted(glob(os.path.join(input_dir, '**', '*.wav'), recursive=True))
    print(f"共发现音频文件: {len(audio_files)}")

    chain_functions = {
        1: apply_pedalboard_chain_1,
        2: apply_pedalboard_chain_2,
        3: apply_pedalboard_chain_3,
        4: apply_pedalboard_chain_4,
    }

    for i, file in enumerate(audio_files):
        base = os.path.basename(file)
        category, drum = parse_category_and_drum(base)
        output_subdir = os.path.join(output_dir, drum, category)
        os.makedirs(output_subdir, exist_ok=True)

        try:
            audio, sr = sf.read(file)
            if audio.ndim > 1:  # 如果是立体声，转单声道
                audio = np.mean(audio, axis=1)

            # 保存原始文件
            orig_path = os.path.join(output_subdir, base)
            sf.write(orig_path, audio, sr, subtype='PCM_16')

            # 处理并保存选择的效果链
            for chain_number in chosen_chains:
                chain_function = chain_functions[chain_number]
                audio_chain = chain_function(audio, sr)
                new_filename = get_chain_filename(file, i + 1, chain_number)
                out_path = os.path.join(output_subdir, new_filename)
                sf.write(out_path, audio_chain, sr, subtype='PCM_16')

                print(f"[✓] 处理完成: {out_path}")
        except Exception as e:
            print(f"[✗] 处理失败 {file}，原因: {e}")

if __name__ == "__main__":
    input_dir = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/Linn_RawDataNew"         # 修改为你的输入路径
    output_dir = "/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/Linn_PedalData"   # 修改为你的输出路径
    chosen_chains = [1, 2, 3, 4]       # 选择你想应用的效果链（可以选择多个，最多4个）

    process_all_with_pedalboard(input_dir, output_dir, chosen_chains)
