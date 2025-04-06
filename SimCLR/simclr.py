import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader


class SimCLRDataTransform:
    """
    为音频MFCC特征实现SimCLR所需的数据增强
    """

    def __init__(self, base_transforms=None, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        augmented_samples = []

        for _ in range(self.n_views):
            # 应用随机增强
            aug_x = self._augment_mfcc(x)

            if self.base_transforms is not None:
                aug_x = self.base_transforms(aug_x)

            augmented_samples.append(aug_x)

        return augmented_samples

    def _augment_mfcc(self, mfcc):
        # Current version randomly chooses ONE augmentation
        # Try applying MULTIPLE augmentations with varying probabilities
        mfcc_np = mfcc.numpy() if isinstance(mfcc, torch.Tensor) else mfcc.copy()

        # Apply pitch shift (simulation for MFCC)
        if np.random.random() < 0.5:
            shift = np.random.uniform(-2, 2)
            # Shift MFCC along frequency axis slightly
            if shift > 0:
                mfcc_np = np.pad(mfcc_np, ((0, 0), (0, 1)), mode='constant')[:, 1:]
            else:
                mfcc_np = np.pad(mfcc_np, ((0, 0), (1, 0)), mode='constant')[:, :-1]

        # Apply time masking with probability
        if np.random.random() < 0.7:
            t_width = np.random.randint(1, max(2, mfcc_np.shape[1] // 5))
            t_start = np.random.randint(0, mfcc_np.shape[1] - t_width)
            mfcc_np[:, t_start:t_start + t_width] = 0

        # Apply frequency masking with probability
        if np.random.random() < 0.7:
            f_width = np.random.randint(1, max(2, mfcc_np.shape[0] // 5))
            f_start = np.random.randint(0, mfcc_np.shape[0] - f_width)
            mfcc_np[f_start:f_start + f_width, :] = 0

        # Apply milder noise with probability
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.05, mfcc_np.shape)
            mfcc_np = mfcc_np + noise

        return torch.tensor(mfcc_np, dtype=torch.float32) if isinstance(mfcc, torch.Tensor) else mfcc_np


class DrumMFCCSimCLRDataset(Dataset):
    """适用于SimCLR的鼓声MFCC数据集"""

    def __init__(self, mfccs, transform=None):
        self.mfccs = mfccs
        self.transform = transform

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        mfcc = self.mfccs[idx]

        if self.transform is not None:
            views = self.transform(mfcc)
            return views

        return mfcc


class NT_Xent(nn.Module):
    """
    改进的归一化温度-缩放交叉熵损失函数
    """

    def __init__(self, batch_size, temperature=0.1, device='cpu'):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self, batch_size):
        # 二维的布尔mask，支持动态批次大小
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        # 设置对角线为False（自身的相似度）
        mask.fill_diagonal_(False)
        # 标记正样本对的位置
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        """
        Args:
            z_i, z_j: 两个视角下的特征表示 [batch_size, dim]
        """
        # 获取实际批次大小
        batch_size = z_i.size(0)

        # 创建适合当前批次大小的mask
        mask = self._get_correlated_mask(batch_size).to(self.device)

        # 计算批次内所有样本对之间的相似度
        p = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]

        # 计算相似度矩阵
        sim = self.similarity_f(p.unsqueeze(1), p.unsqueeze(0)) / self.temperature

        # 只考虑负样本对的相似度
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # 正样本对
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)

        # 负样本对 - 应用mask以仅选择负样本
        negative_samples = sim[mask].reshape(2 * batch_size, -1)

        # 标签总是指向正样本
        labels = torch.zeros(2 * batch_size, device=self.device).long()

        # 正样本和负样本连接在一起
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        loss = self.criterion(logits, labels)
        loss /= (2 * batch_size)

        return loss


class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_dim=512, out_dim=128):
        super(SimCLRProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.projection(x)


class DrumClassifierWithSimCLR(nn.Module):
    """
    集成了SimCLR预训练的鼓声分类器
    """

    def __init__(self, base_encoder, projection_head=None, projection_dim=128):
        super(DrumClassifierWithSimCLR, self).__init__()
        self.encoder = base_encoder

        # 提取encoder的最后一层特征维度
        if not projection_head:
            # 假设base_encoder的最后一个线性层是fc_shared
            in_features = self.encoder.fc_shared.out_features
            self.projection_head = SimCLRProjectionHead(in_features, out_dim=projection_dim)
        else:
            self.projection_head = projection_head

    def forward(self, x):
        # 获取基础编码器的特征
        features = self._get_features(x)

        # 通过投影头得到用于对比学习的表示
        projections = self.projection_head(features)

        return projections

    def _get_features(self, x):
        """提取特征，不包括分类头"""
        # 确保输入有四个维度: [batch_size, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 复用DrumClassifier的特征提取部分，但不包括分类头
        x = self.encoder.pool(F.relu(self.encoder.conv1(x)))
        x = self.encoder.dropout(x)
        x = self.encoder.pool(F.relu(self.encoder.conv2(x)))
        x = self.encoder.dropout(x)
        x = self.encoder.pool(F.relu(self.encoder.conv3(x)))
        x = self.encoder.dropout(x)

        x = x.view(x.size(0), -1)  # 展平
        features = F.relu(self.encoder.fc_shared(x))

        return features


def simclr_train(model, train_loader, optimizer, criterion, epochs=100, device='cpu'):
    """
    使用SimCLR进行预训练

    Args:
        model: SimCLR模型
        train_loader: 数据加载器，提供增强后的样本对
        optimizer: 优化器
        criterion: 对比损失函数
        epochs: 训练轮数
        device: 训练设备
    """
    model.train()
    model = model.to(device)

    # 使用tqdm显示进度条
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    for epoch in range(epochs):
        total_loss = 0

        # 如果可用，使用tqdm作为进度条
        if use_tqdm:
            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        else:
            loop = train_loader
            print(f"Epoch {epoch + 1}/{epochs} 开始训练...")

        batch_count = 0
        for step, views in enumerate(loop):
            optimizer.zero_grad()

            # 检查是否有足够的批次样本
            if len(views[0]) < criterion.batch_size:
                continue

            # 获取两个增强视图
            x_i = views[0].to(device)
            x_j = views[1].to(device)

            # 获取表示
            z_i = model(x_i)
            z_j = model(x_j)

            try:
                # 计算损失
                loss = criterion(z_i, z_j)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # 更新tqdm描述
                if use_tqdm:
                    loop.set_postfix(loss=loss.item())
                elif step % 10 == 0:
                    print(f"  Step: {step}, Loss: {loss.item():.4f}")
            except RuntimeError as e:
                print(f"训练中跳过一个批次，错误: {e}")
                continue

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch: {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        else:
            print(f"警告: Epoch {epoch + 1} 没有处理任何有效批次")

    return model


def prepare_simclr_data(mfcc_tensor, batch_size=32):
    """
    准备SimCLR预训练数据

    Args:
        mfcc_tensor: 包含所有MFCC特征的张量
        batch_size: 批次大小

    Returns:
        data_loader: SimCLR格式的数据加载器
    """
    transform = SimCLRDataTransform(n_views=2)
    dataset = DrumMFCCSimCLRDataset(mfcc_tensor, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return data_loader


def transfer_encoder_weights(simclr_model, classifier_model):
    """
    将预训练的编码器权重转移到分类模型

    Args:
        simclr_model: 预训练的SimCLR模型
        classifier_model: 目标分类器模型

    Returns:
        updated_classifier: 更新了权重的分类器
    """
    # 复制共享部分的权重
    classifier_model.conv1.load_state_dict(simclr_model.encoder.conv1.state_dict())
    classifier_model.conv2.load_state_dict(simclr_model.encoder.conv2.state_dict())
    classifier_model.conv3.load_state_dict(simclr_model.encoder.conv3.state_dict())
    classifier_model.fc_shared.load_state_dict(simclr_model.encoder.fc_shared.state_dict())

    return classifier_model