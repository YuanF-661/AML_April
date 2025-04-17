# 鼓机声音分类器 (Drum Machine Classifier)

这个项目实现了一个鼓机声音分类器，能够同时识别鼓的类型（如Kick、Snare、Cymbal等）和鼓机型号（如808、Linn等）。系统使用MFCC特征提取和CNN神经网络实现分类。现在项目已集成SimCLR自监督学习框架，可以通过无标签数据预训练提升性能。

## 项目结构

```
.
├── config.py                 # 配置文件：数据路径、超参数等
├── dataset.py                # 数据集处理：加载、预处理、MFCC特征提取
├── main.py                   # 推理脚本：用于预测单个音频或批量文件夹
├── main_train.py             # 训练脚本：标准监督学习训练流程
├── main_train_simclr.py      # SimCLR训练脚本：自监督预训练+微调
├── model.py                  # 模型定义：标准CNN分类器
├── model_simclr.py           # 支持SimCLR的增强模型
├── run_simclr_training.py    # SimCLR训练启动脚本
├── run_simclr_training_cpu.py # CPU版本的SimCLR训练启动脚本
├── simclr.py                 # SimCLR框架实现：对比学习、数据增强等
└── train_utils.py            # 训练工具函数：训练循环、评估、可视化
```

## 安装依赖

项目依赖以下Python库：
```
torch
numpy
librosa
scikit-learn
matplotlib
seaborn
tqdm
```

安装命令：
```bash
pip install torch numpy librosa scikit-learn matplotlib seaborn tqdm
```

## 数据集格式

系统期望音频文件采用以下格式命名：`{鼓类型}_{鼓机型号}_{序号}.wav`

例如：`Kick_808_01.wav`, `Snare_Linn_03.wav`

数据集目录路径在 `config.py` 中的 `data_folder` 变量定义。

## 运行流程

### 1. 标准训练 (无SimCLR)

标准训练使用监督学习方法，直接使用标记数据训练分类器：

```bash
python main_train.py
```

### 2. SimCLR预训练 + 微调

SimCLR框架使用两阶段训练：自监督预训练和有监督微调。

#### CPU环境运行：

```bash
python run_simclr_training_cpu.py --simclr_epochs 20 --ft_epochs 10 --batch_size 16
```

参数说明：
- `--simclr_epochs`：SimCLR预训练轮数（默认20）
- `--ft_epochs`：监督微调轮数（默认10）
- `--batch_size`：批次大小（默认16）
- `--temp`：对比学习温度参数（默认0.5）
- `--no_pretrain`：跳过SimCLR预训练阶段
- `--simclr_only`：只进行预训练，不进行微调
- `--data_folder`：可选指定数据集路径

#### GPU环境运行：

```bash
python run_simclr_training.py --simclr_epochs 50 --ft_epochs 20 --batch_size 32
```

### 3. 使用训练好的模型推理

当模型训练完成后，可以使用 `main.py` 对音频文件进行分类：

#### 分析单个音频文件：

```bash
python main.py --audio path/to/audio.wav --model models/DrumClassifier_SimCLR_2025-04-05_v1.pth --metadata models/metadata_simclr.pth
```

#### 批量分析文件夹中的音频：

```bash
python main.py --folder path/to/audio/folder --model models/DrumClassifier_SimCLR_2025-04-05_v1.pth --metadata models/metadata_simclr.pth
```

## SimCLR框架说明

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) 是一种自监督学习框架，通过对比学习引导模型学习更好的特征表示。本项目中的SimCLR实现包括：

1. **数据增强**：对MFCC特征应用频率掩码、时间掩码和噪声添加等增强方法
2. **对比损失**：使用NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失函数
3. **双阶段训练**：先进行自监督预训练，再进行监督微调

## 对比实验

可以通过运行两种训练方式来比较性能差异：

1. 标准监督训练：`python main_train.py`
2. SimCLR增强训练：`python run_simclr_training_cpu.py`

训练后的模型会保存在 `models/` 目录，评估结果会保存在 `results/` 目录。

## 模型文件

训练后生成以下文件：

- `models/DrumClassifier_TypeEnhanced_[日期]_v[版本].pth`：标准模型权重
- `models/DrumClassifier_SimCLR_[日期]_v[版本].pth`：SimCLR微调后的模型权重 
- `models/DrumClassifier_SimCLR_pretrained_[日期].pth`：SimCLR预训练模型权重
- `models/metadata.pth`：标准模型的元数据
- `models/metadata_simclr.pth`：SimCLR模型的元数据

## 可视化结果

训练过程会生成以下可视化结果，保存在 `results/` 目录：

- `training_loss.png`：训练和验证损失曲线
- `validation_accuracy.png`：验证集上的准确率曲线
- `type_confusion_matrix.png`：鼓类型混淆矩阵
- `machine_confusion_matrix.png`：鼓机型号混淆矩阵

## 注意事项

1. SimCLR训练在GPU上效果更好，但提供了CPU版本以兼容无GPU环境
2. 较大的批次大小对SimCLR性能至关重要，但在CPU上可能需要降低批次大小以避免内存问题
3. 推理时，确保使用正确的元数据文件（`metadata.pth`或`metadata_simclr.pth`）
