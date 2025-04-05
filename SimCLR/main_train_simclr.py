import os
import torch
import json
from model import DrumClassifier
from dataset import create_dataset, DrumDataset
from train_utils import train, evaluate_model
from simclr import DrumClassifierWithSimCLR, NT_Xent, prepare_simclr_data, simclr_train, transfer_encoder_weights
import config
from datetime import datetime
from torch.utils.data import random_split

# 从环境变量读取训练参数
SIMCLR_EPOCHS = int(os.environ.get('SIMCLR_EPOCHS', 50))
FT_EPOCHS = int(os.environ.get('FT_EPOCHS', 20))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', config.batch_size))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.5))
SKIP_PRETRAIN = os.environ.get('SKIP_PRETRAIN', '0') == '1'
SIMCLR_ONLY = os.environ.get('SIMCLR_ONLY', '0') == '1'


def main():
    print("===== 开始SimCLR预训练与鼓声识别器训练（使用鼓类型编码增强） =====")

    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 检查数据目录
    print(f"数据目录是否存在: {os.path.exists(config.data_folder)}")
    if os.path.exists(config.data_folder):
        print(f"数据目录内容: {os.listdir(config.data_folder)}")
    else:
        print("错误: 数据目录不存在，请检查配置")
        return

    # 加载数据集
    print("加载数据集...")
    try:
        train_dataset, val_dataset, test_dataset, metadata = create_dataset(
            config.data_folder,
            sample_rate=config.sample_rate,
            target_length=config.target_length
        )
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return

    print(f"数据集加载完成: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本, {len(test_dataset)} 测试样本")
    print(f"识别 {metadata['num_drum_types']} 种鼓类型和 {metadata['num_drum_machines']} 种鼓机型号")

    # 创建带有one-hot编码的数据集
    all_mfccs = []
    all_drum_types = []
    all_drum_machines = []

    # 安全地从原始数据集中提取数据
    try:
        # 从训练集提取
        for i in range(len(train_dataset.indices)):
            mfcc, (drum_type, drum_machine) = train_dataset.dataset[train_dataset.indices[i]]
            all_mfccs.append(mfcc)
            all_drum_types.append(drum_type)
            all_drum_machines.append(drum_machine)

        # 从验证集提取
        for i in range(len(val_dataset.indices)):
            mfcc, (drum_type, drum_machine) = val_dataset.dataset[val_dataset.indices[i]]
            all_mfccs.append(mfcc)
            all_drum_types.append(drum_type)
            all_drum_machines.append(drum_machine)

        # 从测试集提取
        for i in range(len(test_dataset.indices)):
            mfcc, (drum_type, drum_machine) = test_dataset.dataset[test_dataset.indices[i]]
            all_mfccs.append(mfcc)
            all_drum_types.append(drum_type)
            all_drum_machines.append(drum_machine)
    except Exception as e:
        print(f"创建增强数据集时出错: {e}")
        # 如果出错，使用原始数据集继续
        print("使用原始数据集继续训练...")
        use_enhanced_dataset = False
    else:
        # 没有出错时，创建增强数据集
        print(f"成功从原始数据集提取了 {len(all_mfccs)} 个样本")

        try:
            # 创建包含one-hot编码的新数据集
            enhanced_dataset = DrumDataset(
                mfccs=torch.stack(all_mfccs),
                drum_types=torch.tensor(all_drum_types),
                drum_machines=torch.tensor(all_drum_machines),
                num_drum_types=metadata['num_drum_types']
            )

            # 重新分割数据集
            dataset_size = len(enhanced_dataset)
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                enhanced_dataset, [train_size, val_size, test_size]
            )

            print(f"创建了带有one-hot编码的增强数据集，共 {dataset_size} 个样本")
            use_enhanced_dataset = True
        except Exception as e:
            print(f"分割增强数据集时出错: {e}")
            # 如果出错，使用原始数据集继续
            print("使用原始数据集继续训练...")
            use_enhanced_dataset = False

    # 初始化模型
    base_model = DrumClassifier(
        input_channels=1,  # MFCC是单通道
        num_drum_types=metadata['num_drum_types'],
        num_drum_machines=metadata['num_drum_machines'],
        input_height=metadata['input_height'],
        input_width=metadata['input_width']
    )

    # 检测是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # SimCLR预训练阶段
    if not SKIP_PRETRAIN:
        print("\n===== 开始SimCLR自监督预训练 =====")
        # 构建SimCLR模型
        simclr_model = DrumClassifierWithSimCLR(base_model)

        # 创建SimCLR数据加载器
        mfcc_tensor = torch.stack(all_mfccs)
        simclr_loader = prepare_simclr_data(mfcc_tensor, batch_size=BATCH_SIZE)

        # 定义SimCLR训练参数
        simclr_optimizer = torch.optim.Adam(simclr_model.parameters(), lr=config.learning_rate)
        # 传入device参数
        simclr_criterion = NT_Xent(batch_size=BATCH_SIZE, temperature=TEMPERATURE, device=device)

        # 进行SimCLR预训练
        print(f"开始SimCLR预训练，共 {SIMCLR_EPOCHS} 轮...")
        simclr_model = simclr_train(
            simclr_model,
            simclr_loader,
            simclr_optimizer,
            simclr_criterion,
            epochs=SIMCLR_EPOCHS,
            device=device
        )

        # 保存预训练模型
        simclr_save_path = os.path.join('models',
                                        f'DrumClassifier_SimCLR_pretrained_{datetime.now().strftime("%Y%m%d")}.pth')
        torch.save(simclr_model.state_dict(), simclr_save_path)
        print(f"SimCLR预训练模型已保存至: {simclr_save_path}")

        # 只预训练，不微调
        if SIMCLR_ONLY:
            print("\n仅执行SimCLR预训练，跳过监督微调阶段。")
            return

        # 将预训练权重转移到分类器
        print("\n转移预训练权重到分类器模型...")
        # 创建新的分类器模型
        classifier_model = DrumClassifier(
            input_channels=1,
            num_drum_types=metadata['num_drum_types'],
            num_drum_machines=metadata['num_drum_machines'],
            input_height=metadata['input_height'],
            input_width=metadata['input_width']
        )

        # 转移编码器权重
        classifier_model = transfer_encoder_weights(simclr_model, classifier_model)
        print("预训练权重转移完成！")
    else:
        print("\n===== 跳过SimCLR预训练阶段 =====")
        # 直接使用未预训练的模型
        classifier_model = base_model
        simclr_save_path = "没有进行SimCLR预训练"

    # ======== 监督训练阶段 ========
    print("\n===== 开始监督训练阶段 =====")
    # 训练模型
    print("\n使用预训练权重微调模型...")
    train_metrics = train(
        classifier_model,
        train_dataset,
        val_dataset,
        num_epochs=FT_EPOCHS,  # 使用环境变量中的微调轮数
        learning_rate=config.learning_rate * 0.1,  # 降低学习率进行微调
        use_type_onehot=use_enhanced_dataset
    )

    # 评估模型
    print("\n在测试集上评估模型...")
    test_results = evaluate_model(classifier_model, test_dataset, metadata,
                                  use_type_onehot=use_enhanced_dataset)

    # 保存模型
    print("\n保存模型...")

    # 定义保存模型的函数
    def save_model_with_timestamp(model, save_dir="models", base_name="DrumClassifier_SimCLR"):
        os.makedirs(save_dir, exist_ok=True)

        # 获取当前日期
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 查找已有的版本号
        version = 1
        while True:
            filename = f"{base_name}_{date_str}_v{version}.pth"
            save_path = os.path.join(save_dir, filename)
            if not os.path.exists(save_path):
                break
            version += 1

        # 保存模型
        torch.save(model.state_dict(), save_path)
        print(f"✅ 模型已保存：{save_path}")
        return save_path

    save_path = save_model_with_timestamp(classifier_model)

    # 保存类型和鼓机标签映射（将LabelEncoder对象转换为普通的映射字典）
    type_mapping = {}
    for i, cls in enumerate(metadata['type_encoder'].classes_):
        type_mapping[cls] = i

    machine_mapping = {}
    for i, cls in enumerate(metadata['machine_encoder'].classes_):
        machine_mapping[cls] = i

    # 创建元数据字典，用于保存到metadata.pth
    model_metadata = {
        'input_height': metadata['input_height'],
        'input_width': metadata['input_width'],
        'num_drum_types': metadata['num_drum_types'],
        'num_drum_machines': metadata['num_drum_machines'],
        'sample_rate': config.sample_rate,
        'target_length': config.target_length,
        'model_path': save_path,
        'use_type_onehot': use_enhanced_dataset,
        'simclr_pretrained': True,  # 标记使用了SimCLR预训练
        'simclr_model_path': simclr_save_path,  # 保存预训练模型路径
        'drum_type_mapping': type_mapping,
        'drum_machine_mapping': machine_mapping,
        'type_index2label': {int(v): k for k, v in type_mapping.items()},
        'machine_index2label': {int(v): k for k, v in machine_mapping.items()},
        'test_results': {
            'type_accuracy': test_results['type_accuracy'],
            'machine_accuracy': test_results['machine_accuracy']
        }
    }

    # 保存metadata.pth
    metadata_path = os.path.join('models', 'metadata_simclr.pth')
    torch.save(model_metadata, metadata_path)
    print(f"✅ 元数据已保存至: {metadata_path}")

    print(f"模型已保存至: {save_path}")
    print(f"标签映射已保存至models目录")
    print(f"鼓类型识别准确率: {test_results['type_accuracy']:.2f}%")
    print(f"鼓机型号识别准确率: {test_results['machine_accuracy']:.2f}%")
    print("训练完成!")


if __name__ == "__main__":
    main()