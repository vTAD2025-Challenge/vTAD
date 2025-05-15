import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # 导入 tqdm 进度条
import json  # To read the JSON configuration file

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_config(config_file='configs/baseline.json'):
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    config = config_data['train_facodec']
    
    return config


# 数据集类
class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, file_path, is_train=True):
        self.file_path = file_path
        self.is_train = is_train
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        allowed_labels = {'明亮_F', '粗_F', '细_F', '单薄_F', '低沉_F', '干净_F', '厚实_F', '沙哑_F',
            '浑浊_F', '尖锐_F', '圆润_F', '平淡_F', '磁性_F', '干瘪_F', '柔和_F', '沉闷_F',
            '通透_F', '明亮_M', '单薄_M', '磁性_M', '低沉_M', '干净_M', '沉闷_M', '粗_M',
            '浑浊_M', '细_M', '干瘪_M', '厚实_M', '沙哑_M', '平淡_M', '柔和_M', '通透_M',
            '干哑_M', '圆润_M'}  # 这里修改你想要参与的标签

        with open(self.file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                label = parts[0]
                label_tensor = parts[3]

                # 只保留在 allowed_labels 里的数据
                if label not in allowed_labels:
                    continue

                # 读取 .pt 文件
                embedding1 = torch.load(parts[1])  
                embedding2 = torch.load(parts[2])  

                embedding1 = embedding1.squeeze(0)  
                embedding2 = embedding2.squeeze(0)  

                combined_embedding = np.concatenate([embedding1, embedding2], axis=0)


                # 创建标签向量
                label_vector = np.full(34, -1)  # 初始化为-1
                label_index = self.get_label_index(label)
                if label_index != -1:
                    label_vector[label_index] = label_tensor

                self.data.append(combined_embedding)
                self.labels.append(label_vector)

    def get_label_index(self, label):
        labels_list = ['明亮_F', '粗_F', '细_F', '单薄_F', '低沉_F', '干净_F', '厚实_F', '沙哑_F',
            '浑浊_F', '尖锐_F', '圆润_F', '平淡_F', '磁性_F', '干瘪_F', '柔和_F', '沉闷_F',
            '通透_F', '明亮_M', '单薄_M', '磁性_M', '低沉_M', '干净_M', '沉闷_M', '粗_M',
            '浑浊_M', '细_M', '干瘪_M', '厚实_M', '沙哑_M', '平淡_M', '柔和_M', '通透_M',
            '干哑_M', '圆润_M'
        ]
        try:
            return labels_list.index(label)
        except ValueError:
            return -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


class SpeakerEmbeddingModel(nn.Module):
    def __init__(self):
        super(SpeakerEmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        # [添加] 在全连接层后增加一个 Dropout
        self.dropout = nn.Dropout(p=0.3)  # p=0.3为示例，可根据需要调参

        self.fc2 = nn.Linear(128,34)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)               # [添加] 在激活后对 x 执行 dropout
        x = self.sigmoid(self.fc2(x))
        return x

# 训练过程
def train_model(train_file, val_file,checkpoint_dir, model, epochs=10, batch_size=64, learning_rate=0.001, val_epoch=1):

    # 数据加载器
    train_dataset = SpeakerEmbeddingDataset(train_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SpeakerEmbeddingDataset(val_file, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction='none')  # 用于处理-1的二元交叉熵

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # 使用 tqdm 显示训练进度
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
    
            # 创建掩码，筛选出标签为 0 或 1 的部分，-1不参与损失计算
            mask = (labels == 0) | (labels == 1)
            outputs_remain = outputs[mask]
            labels_remain = labels[mask]

            loss = criterion(outputs_remain, labels_remain)
            loss = torch.mean(loss)
    
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
 
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

        if (epoch + 1) % val_epoch == 0:
            save_checkpoint(model, epoch + 1, checkpoint_dir)
            # 验证
            validate_model(val_loader, model, device)

def validate_model(val_loader, model, device):
    # 标签名称列表
    label_names = [
        '明亮_F', '粗_F', '细_F', '单薄_F', '低沉_F', '干净_F', '厚实_F', '沙哑_F',
            '浑浊_F', '尖锐_F', '圆润_F', '平淡_F', '磁性_F', '干瘪_F', '柔和_F', '沉闷_F',
            '通透_F', '明亮_M', '单薄_M', '磁性_M', '低沉_M', '干净_M', '沉闷_M', '粗_M',
            '浑浊_M', '细_M', '干瘪_M', '厚实_M', '沙哑_M', '平淡_M', '柔和_M', '通透_M',
            '干哑_M', '圆润_M'
    ]
    model.eval()
    all_labels = []
    all_preds = []

    # 初始化一个数组来存储每个标签的正确预测数量和总数量
    label_correct = np.zeros(34)  # 假设有34个标签
    label_total = np.zeros(34)

    with torch.no_grad():
        # 使用 tqdm 显示验证进度
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 处理NaN的标签
            mask = labels != -1  # 有效标签的掩码
            pred = (outputs >= 0.5).float()  # 二分类预测，假设阈值为0.5

            # 更新总体标签和预测
            all_labels.append(labels[mask].cpu().numpy())
            all_preds.append(pred[mask].cpu().numpy())

            # 对每个标签计算正确和总数
            for i in range(34):  # 假设有34个标签
                # 获取标签 i 的有效位置
                valid_mask = labels[:, i] != -1
                if valid_mask.sum() > 0:
                    correct_preds = (pred[valid_mask, i] == labels[valid_mask, i]).sum().item()
                    label_correct[i] += correct_preds
                    label_total[i] += valid_mask.sum().item()

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # 计算整个验证集的准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 打印结果
    print(f'Validation Accuracy: {accuracy*100:.2f}')

    # 计算每个标签的精度
    for i in range(34):  # 假设有34个标签
        if label_total[i] > 0:  # 确保标签有有效样本
            label_acc = label_correct[i] / label_total[i]
            print(f'{label_names[i]} Accuracy: {label_acc * 100:.2f}')
        else:
            print(f'{label_names[i]} has no valid samples')


# 保存检查点
def save_checkpoint(model, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

# 主程序
if __name__ == '__main__':
    
    # 使用 get_config 获取配置
    config = get_config()

    set_seed(config['seed'])

    train_file = config['train_path']
    val_file = config['val_path']
    checkpoint_dir=config['checkpoint_dir']

    model = SpeakerEmbeddingModel()
    train_model(
        train_file, 
        val_file, 
        checkpoint_dir,
        model, 
        epochs=config['epochs'], 
        batch_size=config['batch_size'], 
        learning_rate=config['learning_rate'],
        val_epoch=config['val_epoch'],
    )
