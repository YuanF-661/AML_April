import os
import torch
import json
from model import DrumClassifier
from dataset import create_dataset, DrumDataset
from train_utils import train, evaluate_model
import config
from datetime import datetime
from torch.utils.data import random_split


def main():
    print("===== 开始训练鼓声识别器（使用鼓类型编码增强） =====")

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

    # 打印鼓类型和鼓机型号
    print("\n鼓类型:")
    for drum_type, idx in metadata['drum_type_mapping'].items():
        print(f"  {drum_type}: {idx}")

    print("\n鼓机型号:")
    for machine, idx in metadata['drum_machine_mapping'].items():
        print(f"  {machine}: {idx}")

    # 检查数据集大小
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("错误: 一个或多个数据集为空。请检查数据路径和格式。")
        return

    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"训练数据集索引数量: {len(train_dataset.indices)}")

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
    model = DrumClassifier(
        input_channels=1,  # MFCC是单通道
        num_drum_types=metadata['num_drum_types'],
        num_drum_machines=metadata['num_drum_machines'],
        input_height=metadata['input_height'],
        input_width=metadata['input_width']
    )

    # 训练模型
    print("\n开始训练...")
    train_metrics = train(
        model,
        train_dataset,
        val_dataset,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        use_type_onehot=use_enhanced_dataset  # 根据数据集类型决定是否使用one-hot编码
    )

    # 评估模型
    print("\n在测试集上评估模型...")
    test_results = evaluate_model(model, test_dataset, metadata,
                                 use_type_onehot=use_enhanced_dataset)

    # 保存模型
    print("\n保存模型...")

    # 定义保存模型的函数
    def save_model_with_timestamp(model, save_dir="models", base_name="DrumClassifier_TypeEnhanced"):
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

    save_path = save_model_with_timestamp(model)

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
        'drum_type_mapping': type_mapping,  # 直接包含映射字典
        'drum_machine_mapping': machine_mapping,
        'type_index2label': {int(v): k for k, v in type_mapping.items()},  # 添加反向映射
        'machine_index2label': {int(v): k for k, v in machine_mapping.items()},
        'test_results': {  # 保存测试结果
            'type_accuracy': test_results['type_accuracy'],
            'machine_accuracy': test_results['machine_accuracy']
        }
    }

    # 保存metadata.pth
    metadata_path = os.path.join('models', 'metadata.pth')
    torch.save(model_metadata, metadata_path)
    print(f"✅ 元数据已保存至: {metadata_path}")

    # # 为了向后兼容，仍然保存JSON格式的标签映射
    # with open('models/type_label_map.json', 'w') as f:
    #     json.dump(type_mapping, f)
    #
    # with open('models/machine_label_map.json', 'w') as f:
    #     json.dump(machine_mapping, f)
    #
    # with open('models/model_config.json', 'w') as f:
    #     json.dump({
    #         'input_height': metadata['input_height'],
    #         'input_width': metadata['input_width'],
    #         'num_drum_types': metadata['num_drum_types'],
    #         'num_drum_machines': metadata['num_drum_machines'],
    #         'sample_rate': config.sample_rate,
    #         'target_length': config.target_length,
    #         'model_path': save_path,
    #         'type_map_path': 'models/type_label_map.json',
    #         'machine_map_path': 'models/machine_label_map.json',
    #         'use_type_onehot': use_enhanced_dataset
    #     }, f)

    print(f"模型已保存至: {save_path}")
    print(f"标签映射已保存至models目录")
    print(f"鼓类型识别准确率: {test_results['type_accuracy']:.2f}%")
    print(f"鼓机型号识别准确率: {test_results['machine_accuracy']:.2f}%")
    print("训练完成!")


if __name__ == "__main__":
    main()