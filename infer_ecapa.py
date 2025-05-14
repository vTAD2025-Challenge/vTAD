import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm  # 导入 tqdm 进度条
import json  # 导入json模块，用来读取配置文件

# 神经网络模型

class SpeakerEmbeddingModel(nn.Module):
    def __init__(self):
        super(SpeakerEmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)  # Dropout层
        self.fc2 = nn.Linear(128, 34)  # 最终输出34个标签
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# 数据集类

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.labels = []
        self.paths1 = []
        self.paths2 = []
        self.load_data()

    def load_data(self):

        with open(self.file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                label = parts[0]

                embedding1 = torch.load(parts[1])  # (1, 192)
                embedding2 = torch.load(parts[2])  # (1, 192)

                embedding1 = embedding1.squeeze(0)  # 
                embedding2 = embedding2.squeeze(0)  # 

                combined_embedding = np.concatenate([embedding1, embedding2], axis=0)

                self.data.append(combined_embedding)
                self.labels.append(label)  # 保存的是标签字符串
                self.paths1.append(parts[1])  # 保存文件路径
                self.paths2.append(parts[1])  # 保存文件路径

    def get_label_index(self, label):
        labels_list = [
            '明亮_F', '粗_F', '细_F', '单薄_F', '低沉_F', '干净_F', '厚实_F', '沙哑_F',
            '浑浊_F', '尖锐_F', '圆润_F', '平淡_F', '磁性_F', '干瘪_F', '柔和_F', '沉闷_F',
            '通透_F', '明亮_M', '单薄_M', '磁性_M', '低沉_M', '干净_M', '沉闷_M', '粗_M',
            '浑浊_M', '细_M', '干瘪_M', '厚实_M', '沙哑_M', '平淡_M', '柔和_M', '通透_M',
            '干哑_M', '圆润_M'
        ]  # 与训练时候对应的属性顺序需要一致
        try:
            return labels_list.index(label)
        except ValueError:
            return -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx], self.paths1[idx], self.paths2[idx]

# 推理过程

def infer_model(test_file, model_output_path, checkpoint_path, model):
    test_dataset = SpeakerEmbeddingDataset(test_file)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 加载模型
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    with open(model_output_path, 'w') as f:
        with torch.no_grad():
            # 使用 tqdm 显示测试进度
            for inputs, labels, paths1, paths2 in tqdm(test_loader, desc="Inference"):
                inputs = inputs.to(device)
                outputs = model(inputs)

                # 假设阈值为0.5, 将输出大于等于0.5的视为1, 小于的视为0
                pred = (outputs >= 0.5).float()

                # 将结果写入输出文件
                for i in range(len(labels)):
                    label = labels[i]  # 获取标签字符串
                    path1, path2 = paths1[i], paths2[i]  # 获取路径
                    label_index = test_dataset.get_label_index(label)
                    # 只取对应标签的预测概率以及预测结果
                    score = outputs[i][label_index].item()
                    prediction = int(pred[i][label_index].item())
                    f.write(f"{label}|{path1}|{path2}|{score}|{prediction}\n")

if __name__ == '__main__':
    # 读取配置文件
    config_file = 'configs/VADV_baseline.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    test_file = config['infer_ecapa_tdnn']['test_path']
    model_output_path = config['infer_ecapa_tdnn']['model_output']
    checkpoint_path = config['infer_ecapa_tdnn']['checkpoint_path']

    model = SpeakerEmbeddingModel()
    infer_model(test_file, model_output_path, checkpoint_path, model)
