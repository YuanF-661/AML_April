import torch
import torch.nn as nn
import torch.nn.functional as F


class DrumClassifier(nn.Module):
    """
    多任务鼓声分类器：同时识别鼓类型和鼓机型号，
    并利用鼓类型的 one-hot 编码作为额外输入来提高鼓机型号识别

    支持SimCLR预训练
    """

    def __init__(self, input_channels, num_drum_types, num_drum_machines, input_height, input_width):
        super(DrumClassifier, self).__init__()

        # 卷积层 - MFCC特征提取
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # 计算flatten后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            dummy_out = self.pool(F.relu(self.conv1(dummy_input)))
            dummy_out = self.pool(F.relu(self.conv2(dummy_out)))
            dummy_out = self.pool(F.relu(self.conv3(dummy_out)))
            self.flatten_size = dummy_out.view(1, -1).shape[1]

        # 共享特征提取层
        self.fc_shared = nn.Linear(self.flatten_size, 256)

        # 鼓类型分类头
        self.fc_drum_type = nn.Linear(256, num_drum_types)

        # 鼓机型号分类头 - 接收MFCC特征和鼓类型one-hot编码的组合
        self.fc_drum_machine = nn.Linear(256 + num_drum_types, num_drum_machines)

        # 保存维度信息，用于前向传播
        self.num_drum_types = num_drum_types
        self.input_height = input_height
        self.input_width = input_width

    def forward(self, x, drum_type_onehot=None):
        """
        前向传播函数

        Args:
            x: MFCC特征 [batch_size, channels, height, width]
            drum_type_onehot: 可选的鼓类型one-hot编码 [batch_size, num_drum_types]
                              如果为None，则使用模型预测的鼓类型概率
        """
        # 确保输入有四个维度: [batch_size, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 特征提取
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # 展平
        shared_features = F.relu(self.fc_shared(x))

        # 鼓类型预测
        drum_type_logits = self.fc_drum_type(shared_features)

        # 如果没有提供鼓类型的one-hot编码，使用我们的预测
        if drum_type_onehot is None:
            drum_type_probs = F.softmax(drum_type_logits, dim=1)
        else:
            # 使用提供的one-hot编码
            drum_type_probs = drum_type_onehot

        # 将鼓类型概率与共享特征连接，用于鼓机预测
        combined_features = torch.cat([shared_features, drum_type_probs], dim=1)

        # 鼓机型号预测
        drum_machine_logits = self.fc_drum_machine(combined_features)

        return drum_type_logits, drum_machine_logits

    def predict_with_known_type(self, x, drum_type_idx):
        """
        使用已知的鼓类型预测鼓机型号

        Args:
            x: MFCC特征
            drum_type_idx: 鼓类型索引，整数或整数张量

        Returns:
            drum_machine_logits: 鼓机型号预测
        """
        # 确保输入有四个维度
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 特征提取
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # 展平
        shared_features = F.relu(self.fc_shared(x))

        # 创建one-hot编码
        batch_size = x.size(0)

        # 如果drum_type_idx是整数，转换为批次大小的张量
        if isinstance(drum_type_idx, int):
            drum_type_idx = torch.full((batch_size,), drum_type_idx,
                                       dtype=torch.long, device=x.device)

        # 创建one-hot编码
        drum_type_onehot = torch.zeros(batch_size, self.num_drum_types,
                                       device=x.device)
        drum_type_onehot.scatter_(1, drum_type_idx.unsqueeze(1), 1)

        # 组合特征
        combined_features = torch.cat([shared_features, drum_type_onehot], dim=1)

        # 预测鼓机型号
        drum_machine_logits = self.fc_drum_machine(combined_features)

        return drum_machine_logits

    def get_features(self, x):
        """
        提取特征表示，用于SimCLR训练

        Args:
            x: MFCC特征 [batch_size, channels, height, width]

        Returns:
            features: 特征表示 [batch_size, feature_dim]
        """
        # 确保输入有四个维度
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 特征提取
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # 展平
        features = F.relu(self.fc_shared(x))

        return features