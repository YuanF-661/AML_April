import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics
from tqdm import tqdm
import pandas as pd

from model import DrumClassifier
from audio_dataset import create_audio_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='评估知识蒸馏模型性能')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='数据文件夹路径')
    parser.add_argument('--original_model', type=str, required=True,
                        help='原始模型文件路径')
    parser.add_argument('--distilled_model', type=str, required=True,
                        help='蒸馏模型文件路径')
    parser.add_argument('--original_metadata', type=str, required=True,
                        help='原始模型元数据文件路径')
    parser.add_argument('--distilled_metadata', type=str, required=True,
                        help='蒸馏模型元数据文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='输出目录路径')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载元数据
    print("加载元数据...")
    original_metadata = torch.load(args.original_metadata, map_location='cpu')
    distilled_metadata = torch.load(args.distilled_metadata, map_location='cpu')

    # 加载模型
    print("加载原始模型...")
    original_model = load_model(args.original_model, original_metadata)

    print("加载蒸馏模型...")
    distilled_model = load_model(args.distilled_model, distilled_metadata)

    # 加载测试数据集
    print("加载测试数据集...")
    sample_rate = original_metadata.get('sample_rate', 44100)
    target_length = original_metadata.get('target_length', 44100)

    _, _, test_dataset, metadata = create_audio_dataset(
        args.data_folder,
        sample_rate=sample_rate,
        target_length=target_length
    )

    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 评估原始模型
    print("\n评估原始模型...")
    original_results = evaluate_model(
        original_model, test_loader, device,
        use_type_onehot=original_metadata.get('use_type_onehot', True)
    )

    # 评估蒸馏模型
    print("\n评估蒸馏模型...")
    distilled_results = evaluate_model(
        distilled_model, test_loader, device,
        use_type_onehot=distilled_metadata.get('use_type_onehot', True)
    )

    # 打印评估结果
    print("\n===== 原始模型评估结果 =====")
    print(f"测试损失: {original_results['test_loss']:.4f}")
    print(f"鼓类型准确率: {original_results['type_accuracy']:.2f}%")
    print(f"鼓机型号准确率: {original_results['machine_accuracy']:.2f}%")

    print("\n===== 蒸馏模型评估结果 =====")
    print(f"测试损失: {distilled_results['test_loss']:.4f}")
    print(f"鼓类型准确率: {distilled_results['type_accuracy']:.2f}%")
    print(f"鼓机型号准确率: {distilled_results['machine_accuracy']:.2f}%")

    # 计算改进幅度
    type_improvement = distilled_results['type_accuracy'] - original_results['type_accuracy']
    machine_improvement = distilled_results['machine_accuracy'] - original_results['machine_accuracy']

    print("\n===== 模型改进 =====")
    print(f"鼓类型准确率改进: {type_improvement:+.2f}%")
    print(f"鼓机型号准确率改进: {machine_improvement:+.2f}%")

    # 获取标签
    type_labels = [metadata['type_index2label'][i] for i in range(metadata['num_drum_types'])]
    machine_labels = [metadata['machine_index2label'][i] for i in range(metadata['num_drum_machines'])]

    # 绘制混淆矩阵
    print("\n生成混淆矩阵...")
    plot_confusion_matrices(
        distilled_results['type_cm'],
        distilled_results['machine_cm'],
        type_labels,
        machine_labels,
        args.output_dir
    )

    # 绘制置信度直方图
    print("生成置信度直方图...")
    plot_confidence_histograms(distilled_results['sample_confidences'], args.output_dir)

    # 绘制比较图表
    print("生成比较图表...")
    plot_comparative_metrics(original_results, distilled_results, args.output_dir)

    print(f"\n评估完成! 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()


def load_model(model_path, metadata):
    """加载训练好的模型"""
    model = DrumClassifier(
        input_channels=1,
        num_drum_types=metadata['num_drum_types'],
        num_drum_machines=metadata['num_drum_machines'],
        input_height=metadata['input_height'],
        input_width=metadata['input_width']
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


def evaluate_model(model, test_loader, device, use_type_onehot=True):
    """评估模型在测试集上的性能"""
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    type_correct = 0
    machine_correct = 0
    total = 0
    test_loss = 0.0

    # 收集预测和真实标签，用于计算混淆矩阵和分类报告
    all_type_preds = []
    all_type_labels = []
    all_machine_preds = []
    all_machine_labels = []

    # 收集样本级别的置信度
    sample_confidences = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            # 处理批量数据
            if len(batch) == 3:
                mfccs, waveforms, (type_labels, machine_labels, type_onehot) = batch
            else:
                raise ValueError(f"不支持的批量格式: {len(batch)}")

            # 数据移到设备
            mfccs = mfccs.to(device)
            type_labels = type_labels.to(device)
            machine_labels = machine_labels.to(device)

            # 前向传播
            if use_type_onehot:
                type_onehot = type_onehot.to(device)
                type_outputs, machine_outputs = model(mfccs, type_onehot)
            else:
                type_outputs, machine_outputs = model(mfccs)

            # 计算损失
            type_loss = criterion(type_outputs, type_labels)
            machine_loss = criterion(machine_outputs, machine_labels)
            loss = type_loss + machine_loss

            test_loss += loss.item()

            # 计算预测和置信度
            type_probs = torch.softmax(type_outputs, dim=1)
            machine_probs = torch.softmax(machine_outputs, dim=1)

            _, type_preds = torch.max(type_probs, 1)
            _, machine_preds = torch.max(machine_probs, 1)

            # 获取每个样本的预测置信度
            for i in range(type_probs.size(0)):
                # 样本的鼓类型置信度
                type_confidence = type_probs[i, type_preds[i]].item()
                # 样本的鼓机型号置信度
                machine_confidence = machine_probs[i, machine_preds[i]].item()
                # 样本预测是否正确
                type_correct_pred = (type_preds[i] == type_labels[i]).item()
                machine_correct_pred = (machine_preds[i] == machine_labels[i]).item()

                sample_confidences.append({
                    'type_confidence': type_confidence * 100,
                    'machine_confidence': machine_confidence * 100,
                    'type_correct': type_correct_pred,
                    'machine_correct': machine_correct_pred
                })

            # 统计正确预测的数量
            type_correct += (type_preds == type_labels).sum().item()
            machine_correct += (machine_preds == machine_labels).sum().item()
            total += type_labels.size(0)

            # 收集预测和标签
            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(type_labels.cpu().numpy())
            all_machine_preds.extend(machine_preds.cpu().numpy())
            all_machine_labels.extend(machine_labels.cpu().numpy())

    # 计算准确率和损失
    test_loss /= len(test_loader)
    type_accuracy = 100.0 * type_correct / total
    machine_accuracy = 100.0 * machine_correct / total

    # 计算精确率、召回率、F1分数
    type_report = sklearn.metrics.classification_report(
        all_type_labels, all_type_preds, output_dict=True
    )

    machine_report = sklearn.metrics.classification_report(
        all_machine_labels, all_machine_preds, output_dict=True
    )

    # 计算混淆矩阵
    type_cm = sklearn.metrics.confusion_matrix(
        all_type_labels, all_type_preds
    )

    machine_cm = sklearn.metrics.confusion_matrix(
        all_machine_labels, all_machine_preds
    )

    return {
        'test_loss': test_loss,
        'type_accuracy': type_accuracy,
        'machine_accuracy': machine_accuracy,
        'type_report': type_report,
        'machine_report': machine_report,
        'type_cm': type_cm,
        'machine_cm': machine_cm,
        'sample_confidences': sample_confidences
    }


def plot_confusion_matrices(type_cm, machine_cm, type_labels, machine_labels, output_dir):
    """绘制鼓类型和鼓机型号的混淆矩阵"""
    # 绘制鼓类型混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(type_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=type_labels, yticklabels=type_labels)
    plt.xlabel('预测类型')
    plt.ylabel('真实类型')
    plt.title('鼓类型混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'type_confusion_matrix.png'))
    plt.close()

    # 绘制鼓机型号混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(machine_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=machine_labels, yticklabels=machine_labels)
    plt.xlabel('预测型号')
    plt.ylabel('真实型号')
    plt.title('鼓机型号混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'machine_confusion_matrix.png'))
    plt.close()


def plot_confidence_histograms(sample_confidences, output_dir):
    """绘制置信度直方图"""
    # 将置信度数据转换为DataFrame
    df = pd.DataFrame(sample_confidences)

    # 绘制鼓类型置信度直方图
    plt.figure(figsize=(10, 6))
    # 绘制正确和错误预测的置信度分布
    correct_mask = df['type_correct'] == 1

    plt.hist(df.loc[correct_mask, 'type_confidence'], alpha=0.7, bins=20,
             label='正确预测', color='green')
    plt.hist(df.loc[~correct_mask, 'type_confidence'], alpha=0.7, bins=20,
             label='错误预测', color='red')

    plt.xlabel('置信度 (%)')
    plt.ylabel('样本数量')
    plt.title('鼓类型预测置信度分布')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'type_confidence_histogram.png'))
    plt.close()

    # 绘制鼓机型号置信度直方图
    plt.figure(figsize=(10, 6))
    correct_mask = df['machine_correct'] == 1

    plt.hist(df.loc[correct_mask, 'machine_confidence'], alpha=0.7, bins=20,
             label='正确预测', color='green')
    plt.hist(df.loc[~correct_mask, 'machine_confidence'], alpha=0.7, bins=20,
             label='错误预测', color='red')

    plt.xlabel('置信度 (%)')
    plt.ylabel('样本数量')
    plt.title('鼓机型号预测置信度分布')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'machine_confidence_histogram.png'))
    plt.close()


def plot_comparative_metrics(original_results, distilled_results, output_dir):
    """绘制原始模型和蒸馏模型的性能比较"""
    # 准备比较数据
    metrics = {
        'type_accuracy': '鼓类型准确率 (%)',
        'machine_accuracy': '鼓机型号准确率 (%)'
    }

    # 创建条形图比较准确率
    plt.figure(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    original_values = [original_results[key] for key in metrics.keys()]
    distilled_values = [distilled_results[key] for key in metrics.keys()]

    plt.bar(x - width / 2, original_values, width, label='原始模型')
    plt.bar(x + width / 2, distilled_values, width, label='蒸馏模型')

    plt.xlabel('评估指标')
    plt.ylabel('准确率 (%)')
    plt.title('原始模型与蒸馏模型性能比较')
    plt.xticks(x, list(metrics.values()))
    plt.legend()

    # 添加数值标签
    for i, v in enumerate(original_values):
        plt.text(i - width / 2, v + 1, f'{v:.2f}%', ha='center')

    for i, v in enumerate(distilled_values):
        plt.text(i + width / 2, v + 1, f'{v:.2f}%', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()

    # 创建雷达图比较各类分类的F1分数
    fig = plt.figure(figsize=(12, 8))

    # 获取鼓类型的F1分数
    original_type_f1 = {k: v['f1-score'] for k, v in original_results['type_report'].items()
                        if k not in ['accuracy', 'macro avg', 'weighted avg']}
    distilled_type_f1 = {k: v['f1-score'] for k, v in distilled_results['type_report'].items()
                         if k not in ['accuracy', 'macro avg', 'weighted avg']}

    # 确保两个字典有相同的键
    common_keys = set(original_type_f1.keys()) & set(distilled_type_f1.keys())
    categories = list(common_keys)

    # 如果类别太多，可以只选择一部分
    if len(categories) > 10:
        categories = sorted(categories)[:10]

    N = len(categories)

    # 数据准备
    original_values = [original_type_f1[cat] for cat in categories]
    distilled_values = [distilled_type_f1[cat] for cat in categories]

    # 角度计算
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图

    # 添加原始模型与蒸馏模型的数据
    original_values += original_values[:1]
    distilled_values += distilled_values[:1]

    # 绘制雷达图
    ax = fig.add_subplot(111, polar=True)

    plt.xticks(angles[:-1], categories, color='grey', size=10)

    # 绘制原始模型数据
    ax.plot(angles, original_values, 'o-', linewidth=2, label='原始模型')
    ax.fill(angles, original_values, alpha=0.25)

    # 绘制蒸馏模型数据
    ax.plot(angles, distilled_values, 'o-', linewidth=2, label='蒸馏模型')
    ax.fill(angles, distilled_values, alpha=0.25)

    plt.title('各鼓类型的F1分数比较')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_radar.png'))
    plt.close()