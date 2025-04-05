import os
import torch
import librosa
import numpy as np
import argparse
import json
from model import DrumClassifier
from dataset import preprocess, adjust_length
import config
import numpy
import csv
from tqdm import tqdm  # 用于显示进度条
from datetime import datetime

# 导入所需的序列化安全类
import torch.serialization
from sklearn.preprocessing import LabelEncoder

# 将LabelEncoder和numpy scalar添加到安全列表
torch.serialization.add_safe_globals([LabelEncoder, numpy.core.multiarray.scalar])


def extract_labels_from_filename(filename):
    """从文件名中提取真实标签
    例如：Snare_808_10.wav -> drum_type=Snare, drum_machine=808
    """
    # 移除文件扩展名
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # 按下划线分割
    parts = base_name.split('_')

    # 确保至少有两个部分
    if len(parts) >= 2:
        drum_type = parts[0]
        drum_machine = parts[1]
        return drum_type, drum_machine
    else:
        print(f"警告: 无法从文件名 {filename} 中提取标签")
        return None, None


def load_model(model_path, metadata):
    """使用元数据加载训练好的模型"""
    # 初始化模型
    model = DrumClassifier(
        input_channels=1,
        num_drum_types=metadata['num_drum_types'],
        num_drum_machines=metadata['num_drum_machines'],
        input_height=metadata['input_height'],
        input_width=metadata['input_width']
    )

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def prepare_audio_features(audio_file, sample_rate=22050, target_length=22050):
    """从音频文件中提取特征"""
    try:
        audio_data, _ = librosa.load(audio_file, sr=sample_rate)
        adjusted_audio = adjust_length(audio_data, target_length)
        mfcc_features = preprocess(adjusted_audio, sample_rate)
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
        return mfcc_tensor
    except Exception as e:
        print(f"处理音频文件时出错: {e}")
        return None


def predict_drum(model, audio_tensor, type_index2label, machine_index2label, use_type_onehot=False, known_type=None):
    """使用模型预测鼓类型和鼓机型号"""
    with torch.no_grad():
        # 前向传播
        if known_type is not None and use_type_onehot:
            # 只预测鼓机型号，使用已知的鼓类型
            machine_outputs = model.predict_with_known_type(audio_tensor, known_type)
            predicted_type_idx = known_type
            type_confidence = 100.0  # 置信度100%，因为是用户指定的

            # 获取鼓机型号预测
            machine_probs = torch.nn.functional.softmax(machine_outputs, dim=1)[0]
            predicted_machine_idx = torch.argmax(machine_probs).item()
            machine_confidence = machine_probs[predicted_machine_idx].item() * 100
            predicted_machine = machine_index2label[predicted_machine_idx]

            # 只获取鼓机型号的Top-3
            top3_machine_values, top3_machine_indices = torch.topk(machine_probs, min(3, len(machine_probs)))
            top3_machines = [(machine_index2label[idx.item()], val.item() * 100)
                             for idx, val in zip(top3_machine_indices, top3_machine_values)]

            # 鼓类型的Top-3就是单一类型
            predicted_type = type_index2label[predicted_type_idx]
            top3_types = [(predicted_type, 100.0)]
        else:
            # 标准预测，同时预测鼓类型和鼓机型号
            type_outputs, machine_outputs = model(audio_tensor)

            # 获取鼓类型预测
            type_probs = torch.nn.functional.softmax(type_outputs, dim=1)[0]
            predicted_type_idx = torch.argmax(type_probs).item()
            type_confidence = type_probs[predicted_type_idx].item() * 100
            predicted_type = type_index2label[predicted_type_idx]

            # 获取前三可能的鼓类型
            top3_type_values, top3_type_indices = torch.topk(type_probs, min(3, len(type_probs)))
            top3_types = [(type_index2label[idx.item()], val.item() * 100)
                          for idx, val in zip(top3_type_indices, top3_type_values)]

            # 获取鼓机型号预测
            machine_probs = torch.nn.functional.softmax(machine_outputs, dim=1)[0]
            predicted_machine_idx = torch.argmax(machine_probs).item()
            machine_confidence = machine_probs[predicted_machine_idx].item() * 100
            predicted_machine = machine_index2label[predicted_machine_idx]

            # 获取前三可能的鼓机型号
            top3_machine_values, top3_machine_indices = torch.topk(machine_probs, min(3, len(machine_probs)))
            top3_machines = [(machine_index2label[idx.item()], val.item() * 100)
                             for idx, val in zip(top3_machine_indices, top3_machine_values)]

        return {
            'drum_type': predicted_type,
            'type_confidence': type_confidence,
            'top3_types': top3_types,
            'drum_machine': predicted_machine,
            'machine_confidence': machine_confidence,
            'top3_machines': top3_machines
        }


def process_audio_folder(folder_path, model, metadata, type_index2label, machine_index2label, known_type=None):
    """处理文件夹中的所有音频文件并输出到单个CSV文件"""
    # 支持的音频文件扩展名
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif']

    # 获取文件夹中所有音频文件
    audio_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in audio_extensions:
            audio_files.append(file_path)

    if not audio_files:
        print(f"错误: 文件夹 {folder_path} 中没有找到支持的音频文件")
        return

    print(f"找到 {len(audio_files)} 个音频文件，开始处理...")

    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(folder_path), "drum_analysis_results")
    os.makedirs(results_dir, exist_ok=True)

    # 创建包含时间戳的CSV文件名
    timestamp = datetime.now().strftime("%m%d_%H%M")
    folder_name = os.path.basename(os.path.normpath(folder_path))
    csv_path = os.path.join(results_dir, f"SimCLR_{timestamp}.csv")

    # 定义CSV字段名
    fieldnames = [
        '文件名',
        '鼓类型', '类型置信度(%)',
        '鼓机型号', '型号置信度(%)',
        '结果组合',
        '真实鼓类型', '类型预测正确',
        '真实鼓机型号', '机型预测正确',
        '整体预测正确',
        '类型候选1', '类型候选1置信度(%)',
        '类型候选2', '类型候选2置信度(%)',
        '类型候选3', '类型候选3置信度(%)',
        '机型候选1', '机型候选1置信度(%)',
        '机型候选2', '机型候选2置信度(%)',
        '机型候选3', '机型候选3置信度(%)',
        '处理时间', '文件路径'
    ]

    # 统计指标
    total_files = len(audio_files)
    correct_type_count = 0
    correct_machine_count = 0
    correct_both_count = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 使用tqdm显示进度条
        for audio_file in tqdm(audio_files, desc="分析进度"):
            process_start_time = datetime.now()

            # 从文件名提取真实标签
            true_drum_type, true_drum_machine = extract_labels_from_filename(audio_file)

            # 提取音频特征
            audio_tensor = prepare_audio_features(
                audio_file,
                sample_rate=metadata.get('sample_rate', config.sample_rate),
                target_length=metadata.get('target_length', config.target_length)
            )

            if audio_tensor is None:
                print(f"跳过文件 {audio_file}: 无法处理")
                # 记录失败的文件
                writer.writerow({
                    '文件名': os.path.basename(audio_file),
                    '处理时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    '鼓类型': '处理失败',
                    '鼓机型号': '处理失败',
                    '真实鼓类型': true_drum_type or 'Unknown',
                    '真实鼓机型号': true_drum_machine or 'Unknown',
                    '类型预测正确': 'N/A',
                    '机型预测正确': 'N/A',
                    '整体预测正确': 'N/A',
                    '文件路径': audio_file
                })
                continue

            # 确定是否使用one-hot编码增强
            use_type_onehot = metadata.get('use_type_onehot', False)

            # 预测
            results = predict_drum(
                model,
                audio_tensor,
                type_index2label,
                machine_index2label,
                use_type_onehot=use_type_onehot,
                known_type=known_type
            )

            # 验证预测结果
            type_correct = True if true_drum_type and results['drum_type'] == true_drum_type else False
            machine_correct = True if true_drum_machine and results['drum_machine'] == true_drum_machine else False
            both_correct = type_correct and machine_correct

            # 更新统计数据
            if type_correct:
                correct_type_count += 1
            if machine_correct:
                correct_machine_count += 1
            if both_correct:
                correct_both_count += 1

            # 准备CSV行数据
            row_data = {
                '文件名': os.path.basename(audio_file),
                '鼓类型': results['drum_type'],
                '类型置信度(%)': f"{results['type_confidence']:.2f}",
                '鼓机型号': results['drum_machine'],
                '型号置信度(%)': f"{results['machine_confidence']:.2f}",
                '结果组合': f"{results['drum_type']}_{results['drum_machine']}",
                '真实鼓类型': true_drum_type or 'Unknown',
                '类型预测正确': '是' if type_correct else '否',
                '真实鼓机型号': true_drum_machine or 'Unknown',
                '机型预测正确': '是' if machine_correct else '否',
                '整体预测正确': '是' if both_correct else '否',
                '处理时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                '文件路径': audio_file
            }

            # 添加Top3候选项（类型）
            for i in range(min(3, len(results['top3_types']))):
                drum_type, conf = results['top3_types'][i]
                row_data[f'类型候选{i + 1}'] = drum_type
                row_data[f'类型候选{i + 1}置信度(%)'] = f"{conf:.2f}"

            # 补全缺少的类型候选项
            for i in range(len(results['top3_types']), 3):
                row_data[f'类型候选{i + 1}'] = ""
                row_data[f'类型候选{i + 1}置信度(%)'] = ""

            # 添加Top3候选项（机型）
            for i in range(min(3, len(results['top3_machines']))):
                machine, conf = results['top3_machines'][i]
                row_data[f'机型候选{i + 1}'] = machine
                row_data[f'机型候选{i + 1}置信度(%)'] = f"{conf:.2f}"

            # 补全缺少的机型候选项
            for i in range(len(results['top3_machines']), 3):
                row_data[f'机型候选{i + 1}'] = ""
                row_data[f'机型候选{i + 1}置信度(%)'] = ""

            # 写入CSV
            writer.writerow(row_data)

        # 计算并写入总体准确率
        type_accuracy = (correct_type_count / total_files) * 100 if total_files > 0 else 0
        machine_accuracy = (correct_machine_count / total_files) * 100 if total_files > 0 else 0
        overall_accuracy = (correct_both_count / total_files) * 100 if total_files > 0 else 0

        # 写入摘要行
        writer.writerow({
            '文件名': '===== 摘要 =====',
            '鼓类型': f'类型准确率: {type_accuracy:.2f}%',
            '鼓机型号': f'机型准确率: {machine_accuracy:.2f}%',
            '结果组合': f'整体准确率: {overall_accuracy:.2f}%',
            '真实鼓类型': f'总样本数: {total_files}',
            '类型预测正确': f'正确类型数: {correct_type_count}',
            '真实鼓机型号': '',
            '机型预测正确': f'正确机型数: {correct_machine_count}',
            '整体预测正确': f'完全正确数: {correct_both_count}',
            '文件路径': ''
        })

    print(f"\n分析完成！结果保存在: {csv_path}")
    print(f"\n===== 准确率统计 =====")
    print(f"类型准确率: {type_accuracy:.2f}% ({correct_type_count}/{total_files})")
    print(f"机型准确率: {machine_accuracy:.2f}% ({correct_machine_count}/{total_files})")
    print(f"整体准确率: {overall_accuracy:.2f}% ({correct_both_count}/{total_files})")

    return csv_path


def main():
    parser = argparse.ArgumentParser(description='鼓声识别器 - 批量处理')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio', type=str, help='单个音频文件路径')
    group.add_argument('--folder', type=str, help='包含多个音频文件的文件夹路径')
    parser.add_argument('--model', type=str, default='models/DrumClassifier_SimCLR_2025-04-05_v1.pth',
                        help='模型文件路径')
    parser.add_argument('--metadata', type=str, default='models/metadata_simclr.pth',
                        help='元数据文件路径')
    parser.add_argument('--known_type', type=int, help='已知的鼓类型索引（如果提供）')
    parser.add_argument('--save-csv', action='store_true',
                        help='对单个文件的分析结果也保存为CSV')
    args = parser.parse_args()

    # 检查模型和元数据文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return

    if not os.path.exists(args.metadata):
        print(f"错误: 元数据文件 {args.metadata} 不存在")
        return

    # 加载元数据和模型
    print("加载模型...")
    try:
        # 加载元数据
        metadata = torch.load(args.metadata, weights_only=False)  # 显式设置weights_only=False

        # 如果模型路径是相对路径，使用元数据中的路径
        model_path = args.model
        if os.path.basename(model_path) == model_path and 'model_path' in metadata:
            model_path = metadata['model_path']

        # 加载模型
        model = load_model(model_path, metadata)

        # 获取标签映射
        type_index2label = metadata['type_index2label']
        machine_index2label = metadata['machine_index2label']
    except Exception as e:
        print(f"加载模型或元数据时出错: {e}")
        print("\n尝试备用加载方式...")
        try:
            # 设置更多安全类的备用加载方式
            import numpy
            for module_name in ['numpy', 'numpy.core', 'numpy.core.multiarray']:
                module = __import__(module_name, fromlist=['scalar'])
                if hasattr(module, 'scalar'):
                    torch.serialization.add_safe_globals([module.scalar])

            metadata = torch.load(args.metadata)

            # 如果模型路径是相对路径，使用元数据中的路径
            model_path = args.model
            if os.path.basename(model_path) == model_path and 'model_path' in metadata:
                model_path = metadata['model_path']

            # 加载模型
            model = load_model(model_path, metadata)

            # 获取标签映射
            type_index2label = metadata['type_index2label']
            machine_index2label = metadata['machine_index2label']
        except Exception as e2:
            print(f"备用加载方式也失败: {e2}")
            return

    # 根据参数选择处理单个文件或整个文件夹
    if args.audio:
        # 处理单个音频文件
        if not os.path.exists(args.audio):
            print(f"错误: 音频文件 {args.audio} 不存在")
            return

        # 提取音频特征
        print("处理音频...")
        audio_tensor = prepare_audio_features(
            args.audio,
            sample_rate=metadata.get('sample_rate', config.sample_rate),
            target_length=metadata.get('target_length', config.target_length)
        )

        if audio_tensor is None:
            print("无法处理音频文件，请检查文件格式和内容。")
            return

        # 确定是否使用one-hot编码增强
        use_type_onehot = metadata.get('use_type_onehot', False)

        # 预测
        print("进行识别...")
        results = predict_drum(
            model,
            audio_tensor,
            type_index2label,
            machine_index2label,
            use_type_onehot=use_type_onehot,
            known_type=args.known_type
        )

        # 从文件名提取真实标签
        true_drum_type, true_drum_machine = extract_labels_from_filename(args.audio)

        # 验证预测结果
        type_correct = True if true_drum_type and results['drum_type'] == true_drum_type else False
        machine_correct = True if true_drum_machine and results['drum_machine'] == true_drum_machine else False
        both_correct = type_correct and machine_correct

        # 打印结果
        print("\n===== 鼓声识别结果 =====")
        print(f"音频文件: {args.audio}")

        print(f"\n识别的鼓类型: {results['drum_type']} (置信度: {results['type_confidence']:.2f}%)")
        if true_drum_type:
            print(f"真实的鼓类型: {true_drum_type} (预测{'正确' if type_correct else '错误'})")

        print("可能的鼓类型:")
        for i, (drum_type, conf) in enumerate(results['top3_types'], 1):
            print(f"  {i}. {drum_type}: {conf:.2f}%")

        print(f"\n识别的鼓机型号: {results['drum_machine']} (置信度: {results['machine_confidence']:.2f}%)")
        if true_drum_machine:
            print(f"真实的鼓机型号: {true_drum_machine} (预测{'正确' if machine_correct else '错误'})")

        print("可能的鼓机型号:")
        for i, (machine, conf) in enumerate(results['top3_machines'], 1):
            print(f"  {i}. {machine}: {conf:.2f}%")

        print(f"\n最终识别结果: {results['drum_type']}_{results['drum_machine']}")
        if true_drum_type and true_drum_machine:
            print(f"真实组合: {true_drum_type}_{true_drum_machine}")
            print(f"整体预测: {'正确' if both_correct else '错误'}")

        # 如果指定了保存CSV选项，保存单个文件的结果
        if args.save_csv:
            results_dir = os.path.join(os.path.dirname(args.audio), "drum_analysis_results")
            os.makedirs(results_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.basename(args.audio)
            csv_path = os.path.join(results_dir, f"drum_analysis_{timestamp}_{file_name}.csv")

            fieldnames = [
                '文件名',
                '鼓类型', '类型置信度(%)',
                '鼓机型号', '型号置信度(%)',
                '结果组合',
                '真实鼓类型', '类型预测正确',
                '真实鼓机型号', '机型预测正确',
                '整体预测正确',
                '类型候选1', '类型候选1置信度(%)',
                '类型候选2', '类型候选2置信度(%)',
                '类型候选3', '类型候选3置信度(%)',
                '机型候选1', '机型候选1置信度(%)',
                '机型候选2', '机型候选2置信度(%)',
                '机型候选3', '机型候选3置信度(%)',
                '处理时间', '文件路径'
            ]

            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                row_data = {
                    '文件名': os.path.basename(args.audio),
                    '鼓类型': results['drum_type'],
                    '类型置信度(%)': f"{results['type_confidence']:.2f}",
                    '鼓机型号': results['drum_machine'],
                    '型号置信度(%)': f"{results['machine_confidence']:.2f}",
                    '结果组合': f"{results['drum_type']}_{results['drum_machine']}",
                    '真实鼓类型': true_drum_type or 'Unknown',
                    '类型预测正确': '是' if type_correct else '否',
                    '真实鼓机型号': true_drum_machine or 'Unknown',
                    '机型预测正确': '是' if machine_correct else '否',
                    '整体预测正确': '是' if both_correct else '否',
                    '处理时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    '文件路径': args.audio
                }

                # 添加Top3候选项（类型）
                for i in range(min(3, len(results['top3_types']))):
                    drum_type, conf = results['top3_types'][i]
                    row_data[f'类型候选{i + 1}'] = drum_type
                    row_data[f'类型候选{i + 1}置信度(%)'] = f"{conf:.2f}"

                # 补全缺少的类型候选项
                for i in range(len(results['top3_types']), 3):
                    row_data[f'类型候选{i + 1}'] = ""
                    row_data[f'类型候选{i + 1}置信度(%)'] = ""

                # 添加Top3候选项（机型）
                for i in range(min(3, len(results['top3_machines']))):
                    machine, conf = results['top3_machines'][i]
                    row_data[f'机型候选{i + 1}'] = machine
                    row_data[f'机型候选{i + 1}置信度(%)'] = f"{conf:.2f}"

                # 补全缺少的机型候选项
                for i in range(len(results['top3_machines']), 3):
                    row_data[f'机型候选{i + 1}'] = ""
                    row_data[f'机型候选{i + 1}置信度(%)'] = ""

                writer.writerow(row_data)

            print(f"结果已保存到: {csv_path}")

    elif args.folder:
        # 处理整个文件夹的音频文件
        if not os.path.exists(args.folder) or not os.path.isdir(args.folder):
            print(f"错误: 文件夹 {args.folder} 不存在或不是一个目录")
            return

        csv_path = process_audio_folder(
            args.folder,
            model,
            metadata,
            type_index2label,
            machine_index2label,
            args.known_type
        )

        # 询问是否要打开CSV文件
        if csv_path and os.path.exists(csv_path):
            open_csv = input(f"\n是否要打开CSV结果文件? (y/n): ").strip().lower()
            if open_csv == 'y':
                # 尝试使用默认程序打开CSV文件
                import platform
                import subprocess

                system = platform.system()
                try:
                    if system == 'Darwin':  # macOS
                        subprocess.call(('open', csv_path))
                    elif system == 'Windows':
                        os.startfile(csv_path)
                    else:  # Linux或其他系统
                        subprocess.call(('xdg-open', csv_path))
                    print(f"已打开CSV文件: {csv_path}")
                except Exception as e:
                    print(f"无法自动打开CSV文件: {e}")
                    print(f"请手动打开文件: {csv_path}")


if __name__ == "__main__":
    main()