import torch
import torch.nn as nn
import torch.nn.functional as F


class DrumClassifier(nn.Module):
    """
    多任务鼓声分类器：同时识别鼓类型和鼓机型号
    """

    def __init__(self, input_channels, num_drum_types, num_drum_machines, input_height, input_width):
        super(DrumClassifier, self).__init__()

        # 卷积层
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

        # 鼓机型号分类头
        self.fc_drum_machine = nn.Linear(256 + num_drum_types, num_drum_machines)

    def forward(self, x):
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
        drum_type_probs = F.softmax(drum_type_logits, dim=1)

        # 将鼓类型概率与共享特征连接，用于鼓机预测
        combined_features = torch.cat([shared_features, drum_type_probs], dim=1)

        # 鼓机型号预测
        drum_machine_logits = self.fc_drum_machine(combined_features)

        return drum_type_logits, drum_machine_logits