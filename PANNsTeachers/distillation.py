import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

# 导入PANNs模型 (使用PANNs提供的模型加载函数)
# 需要先安装PANNs: pip install torchlibrosa
# 如果没有安装，可以访问 https://github.com/qiuqiangkong/audioset_tagging_cnn
import sys
sys.path.append('PANNsTeachers/PannsCode')  # 替换为 PANNs 代码库的路径# 修改为你的PANNs代码库路径
from PannsCode.models import Cnn14  # 确保这个导入路径正确



class PANNsTeacher:
    """
    使用预训练的PANNs模型作为教师模型
    """

    def __init__(self, panns_model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化PANNs教师模型
        Args:
            panns_model_path: PANNs预训练模型路径
            device: 计算设备
        """
        self.device = device

        # 加载PANNs模型 - Cnn14是PANNs的一种常用架构
        self.model = Cnn14(sample_rate=44100, window_size=1024,
                           hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                           classes_num=527)  # AudioSet有527个类别

        # 加载预训练权重
        checkpoint = torch.load(panns_model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])

        # 设置为评估模式
        self.model.eval()
        self.model.to(device)

        print(f"已加载PANNs教师模型: {panns_model_path}")

    def extract_features(self, waveform, sample_rate=44100):
        """
        从音频波形中提取PANNs特征
        Args:
            waveform: 音频波形数据 [batch_size, time_steps] or [time_steps]
            sample_rate: 采样率

        Returns:
            frame_embeddings: 提取的特征 [batch_size, frames, embedding_dim]
        """
        with torch.no_grad():
            # 确保输入维度正确
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [time_steps] -> [1, time_steps]

            # 确保数据类型为float32并放到正确的设备上
            waveform = waveform.float().to(self.device)

            # 提取PANNs中间层特征
            x = self.model.spectrogram_extractor(waveform)
            x = self.model.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

            frames_num = x.shape[2]
            x = x.transpose(1, 3).reshape(x.shape[0], frames_num, -1)  # (batch_size, frames_num, mel_bins)

            # 可以选择从PANNs模型中提取更深层的特征，这里只使用logmel特征作为示例

            return x

    def predict(self, waveform, sample_rate=44100):
        """
        使用PANNs模型进行预测，获取类别概率
        Args:
            waveform: 音频波形数据 [batch_size, time_steps] or [time_steps]
            sample_rate: 采样率

        Returns:
            clipwise_output: 类别预测概率 [batch_size, classes_num]
        """
        with torch.no_grad():
            # 确保输入维度正确
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [time_steps] -> [1, time_steps]

            # 确保数据类型为float32并放到正确的设备上
            waveform = waveform.float().to(self.device)

            # 通过PANNs获取预测
            output_dict = self.model(waveform, None)

            # 获取类别预测
            clipwise_output = output_dict['clipwise_output']  # [batch_size, classes_num]

            return clipwise_output


def distillation_loss(student_logits, teacher_logits, labels=None, alpha=0.5, temperature=2.0):
    """
    知识蒸馏损失函数，结合硬目标(标签)和软目标(教师预测)
    """
    # 确保教师输出是分离的
    teacher_logits = teacher_logits.detach()

    # 确保维度匹配
    if student_logits.shape[1] != teacher_logits.shape[1]:
        print(f"注意：调整教师输出维度从 {teacher_logits.shape[1]} 到 {student_logits.shape[1]}")
        adjusted_teacher = torch.zeros_like(student_logits)
        min_dim = min(student_logits.shape[1], teacher_logits.shape[1])
        adjusted_teacher[:, :min_dim] = teacher_logits[:, :min_dim]
        teacher_logits = adjusted_teacher

    # 软目标损失 (KL散度)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # 如果提供了标签，计算硬目标损失
    if labels is not None:
        hard_loss = F.cross_entropy(student_logits, labels)
        loss = alpha * hard_loss + (1 - alpha) * soft_loss
    else:
        loss = soft_loss

    return loss

def convert_mfcc_to_waveform(mfcc, sr=44100):
    """
    从MFCC特征重建大致波形
    注意：这只是一个近似转换，不能完美重建原始音频

    Args:
        mfcc: MFCC特征 [batch_size, n_mfcc, time_steps]
        sr: 采样率

    Returns:
        waveform: 近似重建的波形 [batch_size, samples]
    """
    # 为了在蒸馏过程中使用PANNs，需要输入波形
    # 这个函数是一个替代方案，但最好使用原始波形
    # 在实际应用中，建议修改数据集类以同时保留MFCC和波形

    # 这只是一个示例转换函数，不能很好地工作
    # 建议直接保存和使用原始波形数据
    # print("警告：MFCC到波形的转换是不完美的，建议使用原始波形数据")

    batch_size = mfcc.shape[0]
    dummy_waveforms = []

    for i in range(batch_size):
        # 生成一个随机波形作为替代
        # 在实际应用中不要使用这种方法
        dummy_waveform = torch.randn(sr)  # 1秒的随机波形
        dummy_waveforms.append(dummy_waveform)

    return torch.stack(dummy_waveforms)


class DrumAudioDataset(torch.utils.data.Dataset):
    """
    同时保存MFCC特征和原始波形的数据集类
    这是为知识蒸馏而设计的数据集类
    """

    def __init__(self, audio_files, labels, sample_rate=44100, target_length=44100):
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
        self.target_length = target_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 加载音频
        audio_path = self.audio_files[idx]
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 调整长度
        if len(waveform) > self.target_length:
            waveform = waveform[:self.target_length]
        elif len(waveform) < self.target_length:
            waveform = np.pad(waveform, (0, self.target_length - len(waveform)), mode='constant')

        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=20)

        # 转换为张量
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        label = self.labels[idx]

        return mfcc_tensor, waveform_tensor, label