import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import librosa
from sklearn.preprocessing import LabelEncoder


def adjust_length(audio_data, target_length):
    current_length = len(audio_data)
    if current_length > target_length:
        return audio_data[:target_length]
    elif current_length < target_length:
        return np.pad(audio_data, (0, target_length - current_length), mode='constant')
    return audio_data


def preprocess(audio_data, sample_rate):
    # 检查音频数据是否为全零或存在问题
    if np.all(audio_data == 0) or np.max(np.abs(audio_data)) == 0:
        # 防止除零错误，添加一个极小的噪声
        audio_data = np.random.normal(0, 1e-10, len(audio_data))
    else:
        # 正常归一化
        audio_data = audio_data / np.max(np.abs(audio_data))

    # 确保没有 NaN 或 Inf 值
    audio_data = np.nan_to_num(audio_data)

    # 提取MFCC特征
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    return mfcc_features


class DrumDataset(Dataset):
    def __init__(self, mfccs, drum_types, drum_machines):
        self.mfccs = mfccs
        self.drum_types = drum_types
        self.drum_machines = drum_machines

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        return self.mfccs[idx], (self.drum_types[idx], self.drum_machines[idx])


def create_dataset(data_path, sample_rate=22050, target_length=22050, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    print(f"搜索音频文件: {data_path}")
    mfcc_list, drum_types, drum_machines = [], [], []

    # 方法1: 递归搜索匹配 drum_type_machine_type 格式的文件夹
    processed_count = 0

    # 递归处理所有子文件夹
    for root, dirs, files in os.walk(data_path):
        # 获取当前文件夹名
        dir_name = os.path.basename(root)
        parts = dir_name.split('_')

        # 检查文件夹名是否符合 drum_type_machine_type 格式
        if len(parts) >= 2:
            drum_type = parts[0]  # BD, SD, CB等
            machine_type = parts[1]  # 606, 707, 808, 909等

            # 处理该文件夹下的所有wav文件
            wav_files = [f for f in files if f.lower().endswith('.wav')]
            if wav_files:
                print(f"在文件夹 {dir_name} 中找到 {len(wav_files)} 个WAV文件")

                for wav_file in wav_files:
                    file_path = os.path.join(root, wav_file)
                    try:
                        # 加载音频
                        audio_data, _ = librosa.load(file_path, sr=sample_rate)
                        # 调整长度
                        adjusted_audio = adjust_length(audio_data, target_length)
                        # 提取特征
                        mfcc_features = preprocess(adjusted_audio, sample_rate)

                        mfcc_list.append(torch.tensor(mfcc_features, dtype=torch.float32))
                        drum_types.append(drum_type)
                        drum_machines.append(machine_type)

                        processed_count += 1
                        if processed_count % 50 == 0:
                            print(f"已处理 {processed_count} 个文件...")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

    # 方法2: 如果方法1没有找到任何文件，尝试在根目录中搜索wav文件
    if not mfcc_list:
        print("未在子文件夹中找到匹配的音频文件，尝试直接搜索WAV文件...")
        processed_count = 0

        for root, dirs, files in os.walk(data_path):
            wav_files = [f for f in files if f.lower().endswith('.wav')]
            for wav_file in wav_files:
                # 尝试从文件名中提取信息
                parts = wav_file.split('_')
                if len(parts) >= 2:
                    drum_type = parts[0]
                    machine_type = parts[1]

                    file_path = os.path.join(root, wav_file)
                    try:
                        # 加载音频
                        audio_data, _ = librosa.load(file_path, sr=sample_rate)
                        # 调整长度
                        adjusted_audio = adjust_length(audio_data, target_length)
                        # 提取特征
                        mfcc_features = preprocess(adjusted_audio, sample_rate)

                        mfcc_list.append(torch.tensor(mfcc_features, dtype=torch.float32))
                        drum_types.append(drum_type)
                        drum_machines.append(machine_type)

                        processed_count += 1
                        if processed_count % 50 == 0:
                            print(f"已处理 {processed_count} 个文件...")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

    print(f"总共成功处理了 {len(mfcc_list)} 个音频文件")

    if not mfcc_list:
        raise ValueError("没有找到或成功处理任何有效的音频文件。请检查文件路径和格式。")

    # 编码鼓类型和鼓机型号
    type_encoder = LabelEncoder()
    machine_encoder = LabelEncoder()

    drum_types_encoded = type_encoder.fit_transform(drum_types)
    drum_machines_encoded = machine_encoder.fit_transform(drum_machines)

    # 转换为PyTorch张量
    mfcc_tensor = torch.stack(mfcc_list)
    drum_types_tensor = torch.tensor(drum_types_encoded, dtype=torch.long)
    drum_machines_tensor = torch.tensor(drum_machines_encoded, dtype=torch.long)

    # 创建数据集
    dataset = DrumDataset(mfcc_tensor, drum_types_tensor, drum_machines_tensor)

    # 分割数据集
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 获取MFCC特征尺寸
    input_height, input_width = mfcc_list[0].shape

    # 打印找到的鼓类型和鼓机型号
    print("\n找到的鼓类型:")
    for idx, drum_type in enumerate(type_encoder.classes_):
        print(f"  {drum_type}: {idx}")

    print("\n找到的鼓机型号:")
    for idx, machine_type in enumerate(machine_encoder.classes_):
        print(f"  {machine_type}: {idx}")

    # 返回数据集、编码器和元数据
    metadata = {
        'type_encoder': type_encoder,
        'machine_encoder': machine_encoder,
        'num_drum_types': len(type_encoder.classes_),
        'num_drum_machines': len(machine_encoder.classes_),
        'input_height': input_height,
        'input_width': input_width,
        'drum_type_mapping': dict(zip(type_encoder.classes_, type_encoder.transform(type_encoder.classes_))),
        'drum_machine_mapping': dict(zip(machine_encoder.classes_, machine_encoder.transform(machine_encoder.classes_)))
    }

    return train_dataset, val_dataset, test_dataset, metadata