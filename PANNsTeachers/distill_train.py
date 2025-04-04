import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import librosa

# 导入本地模块
from model import DrumClassifier
from dataset import preprocess, adjust_length, create_dataset
from config import batch_size, sample_rate, target_length, learning_rate, num_epochs
from distillation import PANNsTeacher, distillation_loss, convert_mfcc_to_waveform


# 添加参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='鼓机分类器训练 - 知识蒸馏版')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='数据文件夹路径')
    parser.add_argument('--panns_model', type=str, required=True,
                        help='PANNs预训练模型路径（如Cnn14_mAP=0.431.pth）')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='知识蒸馏中硬目标与软目标的权重平衡参数')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='知识蒸馏中的温度参数')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=num_epochs,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=learning_rate,
                        help='学习率')
    parser.add_argument('--use_raw_audio', action='store_true',
                        help='是否使用原始音频波形进行蒸馏（推荐）')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='模型保存目录')

    return parser.parse_args()


def train_with_distillation(model, train_loader, val_loader, teacher_model,
                            optimizer, device, num_epochs,
                            alpha=0.5, temperature=2.0, use_raw_audio=True):
    """
    使用知识蒸馏训练模型

    Args:
        model: 学生模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        teacher_model: PANNs教师模型
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        alpha: 硬目标和软目标损失的权重平衡参数
        temperature: 软化教师预测的温度参数
        use_raw_audio: 是否使用原始音频波形进行蒸馏

    Returns:
        训练历史记录
    """



    print(f"开始训练，使用知识蒸馏 (alpha={alpha}, temperature={temperature})")
    model.to(device)

    # 在创建学生模型之后添加此检查
    def check_requires_grad(model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"警告: 参数 {name} 不需要梯度！")
                param.requires_grad = True

    # 使用
    check_requires_grad(model)

    # 普通分类损失
    criterion = nn.CrossEntropyLoss()

    # 记录训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_type_acc': [], 'val_type_acc': [],
        'train_machine_acc': [], 'val_machine_acc': [],
        'distill_loss': []
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        distill_loss_sum = 0.0
        type_correct = 0
        machine_correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # 只打印一个 batch，调试 batch 的结构
            # for batch in train_loader:
            #     print(f"\n📦 len(batch): {len(batch)}")
            #
            #     for i, item in enumerate(batch):
            #         print(f"\n🔹 batch[{i}] (type: {type(item)})")
            #
            #         if isinstance(item, torch.Tensor):
            #             print(f"  shape: {item.shape}")
            #             print(f"  values (前2个):\n{item[:2]}")
            #
            #         elif isinstance(item, tuple):
            #             print(f"  ⬇️ 内部 tuple 长度: {len(item)}")
            #             for j, sub_item in enumerate(item):
            #                 print(f"    ▪️ item[{j}] (type: {type(sub_item)})")
            #                 if isinstance(sub_item, torch.Tensor):
            #                     print(f"      shape: {sub_item.shape}")
            #                     print(f"      values (前2个):\n{sub_item[:2]}")
            #                 else:
            #                     print(f"      value: {sub_item}")
            #         else:
            #             print(f"  内容: {item}")
            #
            #     break  # 只看一个 batch，避免刷屏

            # 批量数据处理
            if len(batch) == 2 and not use_raw_audio:
                # 标准格式：(mfcc, (type_labels, machine_labels))
                mfcc, (type_labels, machine_labels) = batch

                # 将MFCC转换为波形（近似）
                waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)

            elif len(batch) == 3 and use_raw_audio:
                # 修改这里，正确处理列表形式的标签
                mfcc, waveform, labels_list = batch

                # 从列表中获取标签
                type_labels = labels_list[0]  # 第一个元素是鼓类型标签
                machine_labels = labels_list[1]  # 第二个元素是鼓机型号标签

                # 检查是否有one-hot编码
                if len(labels_list) > 2:
                    type_onehot = labels_list[2]  # 第三个元素是one-hot编码

            elif len(batch) == 3 and not use_raw_audio:
                # one-hot增强格式：(mfcc, (type_labels, machine_labels, type_onehot))
                mfcc, (type_labels, machine_labels, type_onehot) = batch

                # 将MFCC转换为波形（近似）
                waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)

            elif len(batch) == 4:
                # 包含原始波形和one-hot编码的格式
                # (mfcc, waveform, (type_labels, machine_labels, type_onehot))
                mfcc, waveform, (type_labels, machine_labels, type_onehot) = batch

            else:
                raise ValueError(f"不支持的批量格式: {len(batch)}")

            # 数据移到设备
            mfcc = mfcc.to(device)
            waveform = waveform.to(device)
            type_labels = type_labels.to(device)
            machine_labels = machine_labels.to(device)

            # 先获取鼓类型和鼓机型号的数量
            num_drum_types = len(torch.unique(type_labels))
            num_drum_machines = len(torch.unique(machine_labels))

            # 获取教师模型的预测
            with torch.no_grad():
                teacher_outputs = teacher_model.predict(waveform)

                # 将教师输出转换为可以在计算图中使用但不计算梯度的张量
                teacher_type_logits = teacher_outputs[:, :num_drum_types].clone().detach()
                teacher_machine_logits = teacher_outputs[:, -num_drum_machines:].clone().detach()

                # 确保维度匹配
                if teacher_type_logits.size(1) < num_drum_types:
                    padding = torch.zeros(teacher_type_logits.size(0),
                                          num_drum_types - teacher_type_logits.size(1),
                                          device=device)
                    teacher_type_logits = torch.cat([teacher_type_logits, padding], dim=1)

                if teacher_machine_logits.size(1) < num_drum_machines:
                    padding = torch.zeros(teacher_machine_logits.size(0),
                                          num_drum_machines - teacher_machine_logits.size(1),
                                          device=device)
                    teacher_machine_logits = torch.cat([teacher_machine_logits, padding], dim=1)

            # 前向传播（学生模型）
            if 'type_onehot' in locals():
                type_onehot = type_onehot.to(device)
                student_type_logits, student_machine_logits = model(mfcc, type_onehot)
            else:
                student_type_logits, student_machine_logits = model(mfcc)

            # 计算硬目标损失（常规交叉熵）
            type_ce_loss = criterion(student_type_logits, type_labels)
            machine_ce_loss = criterion(student_machine_logits, machine_labels)
            hard_loss = type_ce_loss + machine_ce_loss

            # 在计算知识蒸馏损失前添加
            # print(
            #     f"批次 {batch_idx}: Student type: {student_type_logits.shape}, Teacher type: {teacher_type_logits.shape}")
            # print(
            #     f"批次 {batch_idx}: Student machine: {student_machine_logits.shape}, Teacher machine: {teacher_machine_logits.shape}")

            # 计算知识蒸馏损失
            type_distill_loss = distillation_loss(
                student_type_logits, teacher_type_logits.detach(),  # 明确detach教师输出
                labels=None, alpha=0, temperature=temperature
            )

            machine_distill_loss = distillation_loss(
                student_machine_logits, teacher_machine_logits.detach(),  # 明确detach教师输出
                labels=None, alpha=0, temperature=temperature
            )

            soft_loss = type_distill_loss + machine_distill_loss

            # 组合硬目标和软目标损失
            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            distill_loss_sum += soft_loss.item()

            _, type_preds = torch.max(student_type_logits, 1)
            _, machine_preds = torch.max(student_machine_logits, 1)

            type_correct += (type_preds == type_labels).sum().item()
            machine_correct += (machine_preds == machine_labels).sum().item()
            total += type_labels.size(0)

            # 显示进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Distill Loss: {soft_loss.item():.4f}")

        print(f"Student type logits shape: {student_type_logits.shape}")
        print(f"Teacher type logits shape: {teacher_type_logits.shape}")

        # 计算平均损失和准确率
        train_loss /= len(train_loader)
        distill_loss_avg = distill_loss_sum / len(train_loader)
        train_type_acc = 100.0 * type_correct / total
        train_machine_acc = 100.0 * machine_correct / total

        # 保存到历史记录
        history['train_loss'].append(train_loss)
        history['distill_loss'].append(distill_loss_avg)
        history['train_type_acc'].append(train_type_acc)
        history['train_machine_acc'].append(train_machine_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_type_correct = 0
        val_machine_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                # 批量数据处理（与训练阶段相同）
                if len(batch) == 2 and not use_raw_audio:
                    mfcc, (type_labels, machine_labels) = batch
                    waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)
                elif len(batch) == 3 and use_raw_audio:
                    # 修改这里，与训练阶段保持一致
                    mfcc, waveform, labels_list = batch
                    type_labels = labels_list[0]
                    machine_labels = labels_list[1]
                    if len(labels_list) > 2:
                        type_onehot = labels_list[2]
                elif len(batch) == 3 and not use_raw_audio:
                    mfcc, (type_labels, machine_labels, type_onehot) = batch
                    waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)
                else:
                    raise ValueError(f"不支持的批量格式: {len(batch)}")

                # 数据移到设备
                mfcc = mfcc.to(device)
                type_labels = type_labels.to(device)
                machine_labels = machine_labels.to(device)

                # 学生模型前向传播
                if 'type_onehot' in locals():
                    type_onehot = type_onehot.to(device)
                    student_type_logits, student_machine_logits = model(mfcc, type_onehot)
                else:
                    student_type_logits, student_machine_logits = model(mfcc)

                # 计算验证损失（仅使用硬目标损失）
                type_loss = criterion(student_type_logits, type_labels)
                machine_loss = criterion(student_machine_logits, machine_labels)
                loss = type_loss + machine_loss

                val_loss += loss.item()

                # 计算准确率
                _, type_preds = torch.max(student_type_logits, 1)
                _, machine_preds = torch.max(student_machine_logits, 1)

                val_type_correct += (type_preds == type_labels).sum().item()
                val_machine_correct += (machine_preds == machine_labels).sum().item()
                val_total += type_labels.size(0)

        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_type_acc = 100.0 * val_type_correct / val_total
        val_machine_acc = 100.0 * val_machine_correct / val_total

        # 保存到历史记录
        history['val_loss'].append(val_loss)
        history['val_type_acc'].append(val_type_acc)
        history['val_machine_acc'].append(val_machine_acc)

        # 打印训练信息
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Distill Loss: {distill_loss_avg:.4f}, "
              f"Train Type Acc: {train_type_acc:.2f}%, "
              f"Train Machine Acc: {train_machine_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Type Acc: {val_type_acc:.2f}%, "
              f"Val Machine Acc: {val_machine_acc:.2f}%")

    return history


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置数据目录
    data_folder = args.data_folder if args.data_folder else None
    if data_folder is None:
        # 如果未提供，尝试从config导入
        from config import data_folder

    print(f"数据目录: {data_folder}")
    if not os.path.exists(data_folder):
        print(f"错误: 数据目录 '{data_folder}' 不存在")
        return

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据集
    print("加载数据集...")
    try:
        # 加载数据集时传入 use_raw_audio 参数
        train_dataset, val_dataset, test_dataset, metadata = create_dataset(
            data_folder, sample_rate=sample_rate, target_length=target_length,
            use_raw_audio=args.use_raw_audio
        )
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    print(f"数据集加载完成: {len(train_dataset)} 训练样本, "
          f"{len(val_dataset)} 验证样本, {len(test_dataset)} 测试样本")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载PANNs教师模型
    print(f"加载PANNs教师模型: {args.panns_model}")
    try:
        teacher_model = PANNsTeacher(args.panns_model, device=device)
    except Exception as e:
        print(f"加载教师模型失败: {e}")
        return

    # 创建学生模型
    print("创建学生模型...")
    model = DrumClassifier(
        input_channels=1,  # MFCC是单通道
        num_drum_types=metadata['num_drum_types'],
        num_drum_machines=metadata['num_drum_machines'],
        input_height=metadata['input_height'],
        input_width=metadata['input_width']
    )

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 使用知识蒸馏训练模型
    history = train_with_distillation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_model=teacher_model,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        alpha=args.alpha,
        temperature=args.temperature,
        use_raw_audio=args.use_raw_audio
    )

    # 绘制训练曲线
    plt.figure(figsize=(12, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot(history['distill_loss'], label='Distillation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 鼓类型准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_type_acc'], label='Train Type Acc')
    plt.plot(history['val_type_acc'], label='Val Type Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Drum Type Accuracy')
    plt.legend()

    # 鼓机型号准确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_machine_acc'], label='Train Machine Acc')
    plt.plot(history['val_machine_acc'], label='Val Machine Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Drum Machine Accuracy')
    plt.legend()

    # 保存图表
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/distillation_training_history.png')

    # 在 main 函数中的保存模型和元数据部分
    # 查找存在的版本号
    def get_next_version(dir_path, prefix):
        version = 1
        while os.path.exists(os.path.join(dir_path, f"{prefix}_v{version}.pth")):
            version += 1
        return version

    # 保存模型
    timestamp = datetime.now().strftime("%Y-%m-%d")
    version = get_next_version(args.save_dir, f"DrumClassifier_Distilled_{timestamp}")
    model_filename = f"DrumClassifier_Distilled_{timestamp}_v{version}.pth"
    model_path = os.path.join(args.save_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    # 保存元数据
    metadata_filename = f"metadata_distilled_{timestamp}_v{version}.pth"
    metadata_path = os.path.join(args.save_dir, metadata_filename)

    # 更新元数据以包含蒸馏信息和版本号
    distill_metadata = metadata.copy()
    distill_metadata.update({
        'model_path': model_path,
        'version': version,
        'distillation': {
            'teacher_model': args.panns_model,
            'alpha': args.alpha,
            'temperature': args.temperature,
            'use_raw_audio': args.use_raw_audio
        }
    })

    torch.save(distill_metadata, metadata_path)

    print(f"模型已保存至: {model_path} (版本: v{version})")
    print(f"元数据已保存至: {metadata_path}")

    # 在测试集上评估模型
    print("\n在测试集上评估模型...")
    model.eval()
    test_loss = 0.0
    test_type_correct = 0
    test_machine_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            # 处理批量数据
            if len(batch) == 2 and not args.use_raw_audio:
                mfcc, (type_labels, machine_labels) = batch
            elif len(batch) == 3 and args.use_raw_audio:
                # 处理包含原始波形的批次
                mfcc, waveform, labels_list = batch
                type_labels = labels_list[0]
                machine_labels = labels_list[1]
                if len(labels_list) > 2:
                    type_onehot = labels_list[2]
            elif len(batch) == 3 and not args.use_raw_audio:
                mfcc, (type_labels, machine_labels, type_onehot) = batch
            else:
                raise ValueError(f"不支持的批量格式: {len(batch)}")

            # 数据移到设备
            mfcc = mfcc.to(device)
            type_labels = type_labels.to(device)
            machine_labels = machine_labels.to(device)

            # 前向传播
            if len(batch) == 3:
                type_onehot = type_onehot.to(device)
                type_outputs, machine_outputs = model(mfcc, type_onehot)
            else:
                type_outputs, machine_outputs = model(mfcc)

            # 计算损失
            criterion = nn.CrossEntropyLoss()
            type_loss = criterion(type_outputs, type_labels)
            machine_loss = criterion(machine_outputs, machine_labels)
            loss = type_loss + machine_loss

            test_loss += loss.item()

            # 计算准确率
            _, type_preds = torch.max(type_outputs, 1)
            _, machine_preds = torch.max(machine_outputs, 1)

            test_type_correct += (type_preds == type_labels).sum().item()
            test_machine_correct += (machine_preds == machine_labels).sum().item()
            test_total += type_labels.size(0)

    # 计算平均测试损失和准确率
    test_loss /= len(test_loader)
    test_type_acc = 100.0 * test_type_correct / test_total
    test_machine_acc = 100.0 * test_machine_correct / test_total

    print(f"测试损失: {test_loss:.4f}")
    print(f"测试鼓类型准确率: {test_type_acc:.2f}%")
    print(f"测试鼓机型号准确率: {test_machine_acc:.2f}%")

    # 比较与未使用知识蒸馏的原始模型的性能差异
    print("\n通过知识蒸馏改进的结果：")
    if 'test_results' in metadata and 'type_accuracy' in metadata['test_results']:
        original_type_acc = metadata['test_results']['type_accuracy']
        original_machine_acc = metadata['test_results']['machine_accuracy']

        type_improvement = test_type_acc - original_type_acc
        machine_improvement = test_machine_acc - original_machine_acc

        print(f"鼓类型准确率: {original_type_acc:.2f}% -> {test_type_acc:.2f}% ({type_improvement:+.2f}%)")
        print(f"鼓机型号准确率: {original_machine_acc:.2f}% -> {test_machine_acc:.2f}% ({machine_improvement:+.2f}%)")
    else:
        print("没有原始模型的测试结果用于比较")

    print("\n知识蒸馏训练完成!")


if __name__ == "__main__":
    main()