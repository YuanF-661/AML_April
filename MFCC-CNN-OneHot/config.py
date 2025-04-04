# 数据集目录
data_folder = '/Users/rifi_2001/Documents/UOE Sem-2/Audio Machine Learning/Audio Datasets/In-Use/TrainingAudio'

# 模型保存路径
model_save_path = 'models/DrumClassifier_TypeEnhanced_latest.pth'
metadata_save_path = 'models/metadata.pth'

# 超参数
learning_rate = 0.001
batch_size = 32
num_epochs = 20
sample_rate = 44100
target_length = 44100  # 约1秒的音频