import os
import sys
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='鼓声识别器 - SimCLR预训练与训练（CPU版本）')
    parser.add_argument('--simclr_epochs', type=int, default=25, help='SimCLR预训练轮数')
    parser.add_argument('--ft_epochs', type=int, default=25, help='监督微调轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--data_folder', type=str, default=None, help='数据集路径')
    parser.add_argument('--no_pretrain', action='store_true', help='跳过SimCLR预训练阶段')
    parser.add_argument('--simclr_only', action='store_true', help='只进行SimCLR预训练，不微调')
    parser.add_argument('--temp', type=float, default=0.5, help='SimCLR对比学习温度参数')
    args = parser.parse_args()

    # 强制使用CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # 修改config参数
    if args.data_folder:
        print(f"使用自定义数据路径: {args.data_folder}")
        # 动态修改config.py中的数据路径
        with open('config.py', 'r') as f:
            config_content = f.read()

        # 替换data_folder变量
        import re
        new_config = re.sub(
            r"data_folder = .*",
            f"data_folder = '{args.data_folder}'",
            config_content
        )

        # 写回配置文件
        with open('config.py', 'w') as f:
            f.write(new_config)

    # 设置环境变量控制训练参数
    os.environ['SIMCLR_EPOCHS'] = str(args.simclr_epochs)
    os.environ['FT_EPOCHS'] = str(args.ft_epochs)
    os.environ['BATCH_SIZE'] = str(args.batch_size)
    os.environ['TEMPERATURE'] = str(args.temp)
    os.environ['SKIP_PRETRAIN'] = '1' if args.no_pretrain else '0'
    os.environ['SIMCLR_ONLY'] = '1' if args.simclr_only else '0'

    # 开始时间
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 使用main_train_simclr.py进行训练
    print("=" * 50)
    print("开始SimCLR预训练与监督微调...")
    print("=" * 50)

    # 导入并执行main_train_simclr模块
    try:
        import main_train_simclr
        main_train_simclr.main()
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print("训练完成!")


if __name__ == "__main__":
    main()