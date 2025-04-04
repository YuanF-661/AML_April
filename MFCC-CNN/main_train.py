import os
import torch
from model import DrumClassifier
from dataset import create_dataset
from train_utils import train, evaluate_model
import config


def main():
    print("===== 开始训练鼓声识别模型 =====")

    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 加载数据集
    print("加载数据集...")
    train_dataset, val_dataset, test_dataset, metadata = create_dataset(
        config.data_folder,
        sample_rate=config.sample_rate,
        target_length=config.target_length
    )

    print(f"数据集加载完成: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本, {len(test_dataset)} 测试样本")
    print(f"识别 {metadata['num_drum_types']} 种鼓类型和 {metadata['num_drum_machines']} 种鼓机型号")

    # 打印鼓类型和鼓机型号
    print("\n鼓类型:")
    for drum_type, idx in metadata['drum_type_mapping'].items():
        print(f"  {drum_type}: {idx}")

    print("\n鼓机型号:")
    for machine, idx in metadata['drum_machine_mapping'].items():
        print(f"  {machine}: {idx}")

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
    metrics = train(
        model,
        train_dataset,
        val_dataset,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate
    )

    # 评估模型
    print("\n在测试集上评估模型...")
    test_results = evaluate_model(model, test_dataset, metadata)

    # 保存模型
    print("\n保存模型...")
    torch.save(model.state_dict(), config.model_save_path)

    # 保存元数据
    torch.save(metadata, config.metadata_save_path)

    print(f"模型和元数据已保存至models目录")
    print(f"鼓类型识别准确率: {test_results['type_accuracy']:.2f}%")
    print(f"鼓机型号识别准确率: {test_results['machine_accuracy']:.2f}%")
    print("训练完成!")


if __name__ == "__main__":
    main()