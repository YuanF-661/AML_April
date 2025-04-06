import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import os
from config import batch_size
import seaborn as sns


def train(model, train_dataset, val_dataset, num_epochs=20, learning_rate=0.001, use_type_onehot=True):
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 添加学习率调度器 - 当验证损失停止改善时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # 学习率减半
        patience=3,  # 等待3个epoch无改善
        verbose=True,
        min_lr=1e-6  # 最小学习率
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 保存训练过程中的指标
    train_losses = []
    val_losses = []
    val_type_accuracies = []
    val_machine_accuracies = []

    # 记录最佳模型
    best_val_loss = float('inf')
    best_model_state = None

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        type_correct = 0
        machine_correct = 0
        total_samples = 0

        # 使用tqdm显示进度条（如果可用）
        try:
            from tqdm import tqdm
            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        except ImportError:
            loop = train_loader
            print(f"Epoch {epoch + 1}/{num_epochs} 开始训练...")

        for batch in loop:
            optimizer.zero_grad()

            # 根据是否使用one-hot编码处理不同的数据格式
            if use_type_onehot and len(batch[1]) == 3:
                # 使用drum_type的one-hot编码
                inputs, (type_labels, machine_labels, type_onehot) = batch

                # 前向传播时传入one-hot编码
                type_outputs, machine_outputs = model(inputs, type_onehot)
            else:
                # 标准格式
                inputs, (type_labels, machine_labels) = batch

                # 标准前向传播
                type_outputs, machine_outputs = model(inputs)

            # 计算损失
            type_loss = criterion(type_outputs, type_labels)
            machine_loss = criterion(machine_outputs, machine_labels)

            # 可以调整权重来平衡两个任务的学习
            loss = 0.5 * type_loss + 0.5 * machine_loss

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            _, type_preds = torch.max(type_outputs, 1)
            _, machine_preds = torch.max(machine_outputs, 1)
            type_correct += (type_preds == type_labels).sum().item()
            machine_correct += (machine_preds == machine_labels).sum().item()
            total_samples += type_labels.size(0)

            # 更新进度条
            if hasattr(loop, 'set_postfix'):
                loop.set_postfix(loss=loss.item(),
                                 type_acc=100 * type_correct / max(1, total_samples),
                                 machine_acc=100 * machine_correct / max(1, total_samples))

        avg_train_loss = total_loss / len(train_loader)
        train_type_accuracy = 100 * type_correct / total_samples
        train_machine_accuracy = 100 * machine_correct / total_samples
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        val_type_correct = 0
        val_machine_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                # 根据是否使用one-hot编码处理不同的数据格式
                if use_type_onehot and len(batch[1]) == 3:
                    # 使用drum_type的one-hot编码
                    inputs, (type_labels, machine_labels, type_onehot) = batch

                    # 验证两种方式：
                    # 1. 使用真实的drum_type的one-hot编码
                    type_outputs, machine_outputs_with_true = model(inputs, type_onehot)

                    # 2. 使用模型预测的drum_type概率
                    _, machine_outputs_with_pred = model(inputs)

                else:
                    # 标准格式
                    inputs, (type_labels, machine_labels) = batch

                    # 标准前向传播
                    type_outputs, machine_outputs_with_pred = model(inputs)
                    machine_outputs_with_true = machine_outputs_with_pred  # 此时二者相同

                # 计算损失 - 使用预测的drum_type概率
                type_loss = criterion(type_outputs, type_labels)
                machine_loss = criterion(machine_outputs_with_pred, machine_labels)
                loss = 0.5 * type_loss + 0.5 * machine_loss

                val_loss += loss.item()

                # 计算准确率 - 使用预测的drum_type的预测
                _, type_preds = torch.max(type_outputs, 1)
                _, machine_preds = torch.max(machine_outputs_with_pred, 1)

                val_type_correct += (type_preds == type_labels).sum().item()
                val_machine_correct += (machine_preds == machine_labels).sum().item()
                val_total += type_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_type_accuracy = 100 * val_type_correct / val_total
        val_machine_accuracy = 100 * val_machine_correct / val_total

        val_losses.append(avg_val_loss)
        val_type_accuracies.append(val_type_accuracy)
        val_machine_accuracies.append(val_machine_accuracy)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f">>> 更新最佳模型 (新的验证损失: {best_val_loss:.4f})")

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"学习率 = {optimizer.param_groups[0]['lr']:.6f}, "
              f"训练损失 = {avg_train_loss:.4f}, "
              f"类型准确率 = {train_type_accuracy:.2f}%, "
              f"机型准确率 = {train_machine_accuracy:.2f}%, "
              f"验证损失 = {avg_val_loss:.4f}, "
              f"验证类型准确率 = {val_type_accuracy:.2f}%, "
              f"验证机型准确率 = {val_machine_accuracy:.2f}%")

    # 训练完成后恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复最佳模型，验证损失: {best_val_loss:.4f}")

    # 绘制训练曲线
    os.makedirs('results', exist_ok=True)

    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.savefig('results/training_loss.png')

    # 准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(val_type_accuracies, label='鼓类型准确率')
    plt.plot(val_machine_accuracies, label='鼓机型号准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.title('验证准确率')
    plt.legend()
    plt.savefig('results/validation_accuracy.png')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_type_accuracies': val_type_accuracies,
        'val_machine_accuracies': val_machine_accuracies,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, test_dataset, metadata, use_type_onehot=True):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    type_correct = 0
    machine_correct = 0
    total = 0

    # 预测和真实标签
    all_type_preds = []
    all_type_labels = []
    all_machine_preds = []
    all_machine_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # 根据是否使用one-hot编码处理不同的数据格式
            if use_type_onehot and len(batch[1]) == 3:
                # 使用drum_type的one-hot编码
                inputs, (type_labels, machine_labels, type_onehot) = batch

                # 前向传播时传入one-hot编码
                type_outputs, machine_outputs = model(inputs, type_onehot)
            else:
                # 标准格式
                inputs, (type_labels, machine_labels) = batch

                # 标准前向传播
                type_outputs, machine_outputs = model(inputs)

            type_loss = criterion(type_outputs, type_labels)
            machine_loss = criterion(machine_outputs, machine_labels)
            loss = type_loss + machine_loss

            test_loss += loss.item()

            _, type_preds = torch.max(type_outputs, 1)
            _, machine_preds = torch.max(machine_outputs, 1)

            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(type_labels.cpu().numpy())
            all_machine_preds.extend(machine_preds.cpu().numpy())
            all_machine_labels.extend(machine_labels.cpu().numpy())

            type_correct += (type_preds == type_labels).sum().item()
            machine_correct += (machine_preds == machine_labels).sum().item()
            total += type_labels.size(0)

    test_loss = test_loss / len(test_loader)
    type_accuracy = 100 * type_correct / total
    machine_accuracy = 100 * machine_correct / total

    # [后续代码保持不变]...

    # 生成混淆矩阵
    type_cm = sklearn.metrics.confusion_matrix(all_type_labels, all_type_preds)
    machine_cm = sklearn.metrics.confusion_matrix(all_machine_labels, all_machine_preds)

    # 获取鼓类型和鼓机名称
    type_encoder = metadata['type_encoder']
    machine_encoder = metadata['machine_encoder']

    type_names = type_encoder.classes_
    machine_names = machine_encoder.classes_

    # 绘制鼓类型混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(type_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=type_names, yticklabels=type_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Drum Type Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/type_confusion_matrix.png')

    # 绘制鼓机混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(machine_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=machine_names, yticklabels=machine_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Drum Machine Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/machine_confusion_matrix.png')

    # 计算分类报告
    type_report = sklearn.metrics.classification_report(
        all_type_labels, all_type_preds,
        target_names=type_names,
        output_dict=True
    )

    machine_report = sklearn.metrics.classification_report(
        all_machine_labels, all_machine_preds,
        target_names=machine_names,
        output_dict=True
    )

    # 打印测试结果
    print("\n===== 测试结果 =====")
    print(f"测试损失: {test_loss:.4f}")
    print(f"鼓类型准确率: {type_accuracy:.2f}%")
    print(f"鼓机型号准确率: {machine_accuracy:.2f}%")

    print("\n鼓类型分类报告:")
    for drum_type, metrics in type_report.items():
        if drum_type not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{drum_type}: Precision: {metrics['precision']:.2f}, "
                  f"Recall: {metrics['recall']:.2f}, "
                  f"F1-Score: {metrics['f1-score']:.2f}")

    print("\n鼓机型号分类报告:")
    for machine, metrics in machine_report.items():
        if machine not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{machine}: Precision: {metrics['precision']:.2f}, "
                  f"Recall: {metrics['recall']:.2f}, "
                  f"F1-Score: {metrics['f1-score']:.2f}")

    return {
        'test_loss': test_loss,
        'type_accuracy': type_accuracy,
        'machine_accuracy': machine_accuracy,
        'type_cm': type_cm,
        'machine_cm': machine_cm,
        'type_report': type_report,
        'machine_report': machine_report
    }