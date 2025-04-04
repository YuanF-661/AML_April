import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import time


def adjust_length(audio_data, target_length):
    """调整音频长度到目标长度"""
    current_length = len(audio_data)
    if current_length > target_length:
        return audio_data[:target_length]
    elif current_length < target_length:
        return np.pad(audio_data, (0, target_length - current_length), mode='constant')
    return audio_data


def extract_mfcc(audio_data, sample_rate, n_mfcc=20):
    """从音频波形提取MFCC特征"""
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
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc_features


class DrumAudioDataset(Dataset):
    """
    同时保存MFCC特征和原始波形的数据集类，专为知识蒸馏设计
    """

    def __init__(self, mfccs=None, waveforms=None, drum_types=None, drum_machines=None,
                 num_drum_types=None, use_type_onehot=True):
        """
        初始化数据集

        Args:
            mfccs: MFCC特征张量列表
            waveforms: 音频波形张量列表
            drum_types: 鼓类型标签张量
            drum_machines: 鼓机型号标签张量
            num_drum_types: 鼓类型总数（用于one-hot编码）
            use_type_onehot: 是否使用one-hot编码增强
        """
        self.mfccs = mfccs
        self.waveforms = waveforms
        self.drum_types = drum_types
        self.drum_machines = drum_machines
        self.num_drum_types = num_drum_types
        self.use_type_onehot = use_type_onehot

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        # 获取MFCC特征
        mfcc = self.mfccs[idx]

        # 获取音频波形
        waveform = self.waveforms[idx]

        # 获取标签
        drum_type = self.drum_types[idx]
        drum_machine = self.drum_machines[idx]

        # 如果需要，创建drum_type的one-hot编码
        if self.use_type_onehot and self.num_drum_types is not None:
            # 创建one-hot编码
            drum_type_onehot = torch.zeros(self.num_drum_types)
            drum_type_onehot[drum_type] = 1
            return mfcc, waveform, (drum_type, drum_machine, drum_type_onehot)

        # 否则，按原样返回
        return mfcc, waveform, (drum_type, drum_machine)


def create_audio_dataset(data_path, sample_rate=44100, target_length=44100,
                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    创建同时包含MFCC特征和原始波形的数据集

    Args:
        data_path: 音频文件的目录路径
        sample_rate: 采样率
        target_length: 目标音频长度
        train_ratio, val_ratio, test_ratio: 训练/验证/测试数据的比例

    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    print(f"搜索音频文件: {data_path}")
    mfcc_list, waveform_list, drum_types, drum_machines = [], [], [], []

    cache_dir = os.path.join(os.path.dirname(data_path), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"audio_cache_{sample_rate}_{target_length}.pt")

    # 尝试加载缓存
    if os.path.exists(cache_file):
        print(f"发现缓存文件，正在加载: {cache_file}")
        try:
            cache_data = torch.load(cache_file)
            mfcc_list = cache_data['mfcc_list']
            waveform_list = cache_data['waveform_list']
            drum_types = cache_data['drum_types']
            drum_machines = cache_data['drum_machines']
            print(f"成功从缓存加载 {len(mfcc_list)} 个样本")

            if len(mfcc_list) > 0:
                # 直接跳到编码部分
                goto_encoding = True
            else:
                goto_encoding = False
                print("缓存为空，重新处理音频文件")
        except Exception as e:
            print(f"加载缓存失败: {e}")
            goto_encoding = False
    else:
        goto_encoding = False
        print("未找到缓存文件，将处理音频文件并创建缓存")

    # 处理音频文件
    if not goto_encoding:
        # 递归处理所有子文件夹
        processed_count = 0
        start_time = time.time()

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
                            audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=True)

                            # 调整长度
                            adjusted_audio = adjust_length(audio_data, target_length)

                            # 提取MFCC特征
                            mfcc_features = extract_mfcc(adjusted_audio, sample_rate)

                            # 保存MFCC和波形
                            mfcc_list.append(torch.tensor(mfcc_features, dtype=torch.float32))
                            waveform_list.append(torch.tensor(adjusted_audio, dtype=torch.float32))
                            drum_types.append(drum_type)
                            drum_machines.append(machine_type)

                            processed_count += 1
                            if processed_count % 50 == 0:
                                elapsed = time.time() - start_time
                                print(f"已处理 {processed_count} 个文件... 耗时: {elapsed:.2f}秒")
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

        # 如果方法1没有找到任何文件，尝试在根目录中搜索wav文件
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
                            audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=True)

                            # 调整长度
                            adjusted_audio = adjust_length(audio_data, target_length)

                            # 提取MFCC特征
                            mfcc_features = extract_mfcc(adjusted_audio, sample_rate)

                            # 保存MFCC和波形
                            mfcc_list.append(torch.tensor(mfcc_features, dtype=torch.float32))
                            waveform_list.append(torch.tensor(adjusted_audio, dtype=torch.float32))
                            drum_types.append(drum_type)
                            drum_machines.append(machine_type)

                            processed_count += 1
                            if processed_count % 50 == 0:
                                elapsed = time.time() - start_time
                                print(f"已处理 {processed_count} 个文件... 耗时: {elapsed:.2f}秒")
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

        print(f"总共成功处理了 {len(mfcc_list)} 个音频文件")

        # 保存缓存
        if len(mfcc_list) > 0:
            print(f"保存处理结果到缓存: {cache_file}")
            torch.save({
                'mfcc_list': mfcc_list,
                'waveform_list': waveform_list,
                'drum_types': drum_types,
                'drum_machines': drum_machines
            }, cache_file)
            print("缓存保存完成")

    if not mfcc_list:
        raise ValueError("没有找到或成功处理任何有效的音频文件。请检查文件路径和格式。")

    # 编码鼓类型和鼓机型号
    type_encoder = LabelEncoder()
    machine_encoder = LabelEncoder()

    drum_types_encoded = type_encoder.fit_transform(drum_types)
    drum_machines_encoded = machine_encoder.fit_transform(drum_machines)

    # 转换为PyTorch张量
    mfcc_tensor = torch.stack(mfcc_list)
    waveform_tensor = torch.stack(waveform_list)
    drum_types_tensor = torch.tensor(drum_types_encoded, dtype=torch.long)
    drum_machines_tensor = torch.tensor(drum_machines_encoded, dtype=torch.long)

    # 创建带有one-hot编码的增强数据集
    num_drum_types = len(type_encoder.classes_)
    dataset = DrumAudioDataset(
        mfccs=mfcc_tensor,
        waveforms=waveform_tensor,
        drum_types=drum_types_tensor,
        drum_machines=drum_machines_tensor,
        num_drum_types=num_drum_types,
        use_type_onehot=True
    )

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
        'drum_machine_mapping': dict(
            zip(machine_encoder.classes_, machine_encoder.transform(machine_encoder.classes_))),
        'sample_rate': sample_rate,
        'target_length': target_length,
        'type_index2label': {int(i): label for i, label in enumerate(type_encoder.classes_)},
        'machine_index2label': {int(i): label for i, label in enumerate(machine_encoder.classes_)}
    }

    return train_dataset, val_dataset, test_dataset, metadata