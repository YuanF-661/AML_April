import os
from collections import defaultdict

def count_and_calculate_ratios(root_folder):
    folder_file_counts = defaultdict(dict)

    # 遍历所有文件夹，找出最底层的子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if not dirnames:  # 最底层文件夹
            rel_path = os.path.relpath(dirpath, root_folder)
            parent_path = os.path.dirname(rel_path)
            folder_name = os.path.basename(rel_path)
            file_count = len([
                f for f in filenames if os.path.isfile(os.path.join(dirpath, f))
            ])
            folder_file_counts[parent_path][folder_name] = file_count

    # 遍历并打印比例
    for parent, children in folder_file_counts.items():
        total = sum(children.values())
        for folder_name, count in children.items():
            ratio = (count / total) * 100 if total > 0 else 0
            full_path = os.path.join(parent, folder_name)
            print(f"{full_path}: {count} file(s), {ratio:.2f}%")


# 使用方式
root_folder = '/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/808_Linn'  # 替换为你的根目录路径
count_and_calculate_ratios(root_folder)
