
# coding: utf-8
# In[1]: 导入必要的库函数

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from utils import read_directory
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
from scipy.io import loadmat,savemat
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# In[2] 加载数据
num_classes=8
height=64
width=64

# 小波时频图---2D-CNN输入
x_train,y_train=read_directory('小波时频33/train_img',height,width,normal=1)
x_valid,y_valid=read_directory('小波时频33/valid_img',height,width,normal=1)

# FFT频域信号--1D-CNN输入
datafft=loadmat('FFT频谱33/FFT.mat')
x_train2=datafft['train_X']
x_valid2=datafft['valid_X']
ss2=StandardScaler().fit(x_train2)
x_train2=ss2.transform(x_train2)
x_valid2=ss2.transform(x_valid2)

x_train2=x_train2.reshape(x_train2.shape[0],1,-1)
x_valid2=x_valid2.reshape(x_valid2.shape[0],1,-1)

# 转换为torch的输入格式
train_features = torch.tensor(x_train).type(torch.FloatTensor)
valid_features = torch.tensor(x_valid).type(torch.FloatTensor)

train_features2 = torch.tensor(x_train2).type(torch.FloatTensor)
valid_features2 = torch.tensor(x_valid2).type(torch.FloatTensor)
#
# train_labels = torch.tensor(y_train).type(torch.LongTensor)
# valid_labels = torch.tensor(y_valid).type(torch.LongTensor)

train_labels = torch.tensor(y_train-1).type(torch.LongTensor)
valid_labels = torch.tensor(y_valid-1).type(torch.LongTensor)
print("Unique labels in y_train:", y_train)
print("Unique labels in y_valid:", y_valid)

# 假设 y_train 和 y_valid 是原始的标签，train_labels 是减1之后的标签
# 恢复标签
# original_train_labels = train_labels + 1
# original_valid_labels = valid_labels + 1

# 打印真实的标签
# print("Original train labels:", original_train_labels)
# print("Original valid labels:", original_valid_labels)

print("train_features.shape:", train_features.shape)  # 打印出 x1 的形状
print("train_features2.shape:", train_features2.shape)  # 打印出 x1 的形状
print("train_labels.shape:", train_labels.shape)  # 打印出 x1 的形状

N=train_features.size(0)

# In[3]: 参数设置
learning_rate = 0.005#学习率
num_epochs = 200#迭代次数
batch_size = 64 #batchsize

# In[4]:
# 模型设置
# CNN-Mamba结构
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mamba import Mamba
#
# class ConvNetWithMamba(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(ConvNetWithMamba, self).__init__()
#
#         # 2D-CNN 输入为64*64*3 的图片
#         self.net1 = nn.Sequential(
#             nn.Conv2d(3, 6, kernel_size=5),    # 64-5+1=60 -> 60*60*6
#             nn.MaxPool2d(kernel_size=2),       # 60/2=30 -> 30*30*6
#             nn.ReLU(),
#             nn.BatchNorm2d(6),
#             nn.Conv2d(6, 16, kernel_size=5),   # 30-5+1=26 -> 26*26*16
#             nn.MaxPool2d(kernel_size=2),       # 26/2=13 -> 13*13*16
#             nn.ReLU(),
#             nn.BatchNorm2d(16)
#         )
#
#         # 1D-CNN 输入1*433FFT频谱信号
#         self.net3 = nn.Sequential(
#             nn.Conv1d(1, 6, kernel_size=5),    # 433-5+1=429 -> 1*429*6
#             nn.MaxPool1d(kernel_size=3),       # 429/3=143 -> 1*143*6
#             nn.ReLU(),
#             nn.BatchNorm1d(6),
#             nn.Conv1d(6, 16, kernel_size=3),   # 143-3+1=141 -> 1*141*16
#             nn.MaxPool1d(kernel_size=3),       # 141/3=47 -> 1*47*16
#             nn.ReLU(),
#             nn.BatchNorm1d(16)
#         )
#
#         # 引入 Mamba 模块
#         self.mamba = Mamba(seq_len=47, d_model=16, state_size=128, device='cuda')
#
#         self.feature_layer = nn.Sequential(
#             nn.Linear(2704 + 752, 120),       # 全连接层
#             nn.ReLU(),
#             nn.Linear(120, 84),
#         )
#         self.classifier = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(84, num_classes),
#         )
#
#     def forward(self, x1, x2):
#         x1 = self.net1(x1)  # 提取图片特征
#         x2 = self.net3(x2)  # 提取 FFT 特征
#
#         # 使用 Mamba 处理 FFT 特征
#         x2 = x2.permute(0, 2, 1)   # 调整维度以匹配 Mamba 输入 (B, L, D)
#         x2 = self.mamba(x2)
#         x2 = x2.view(-1, 752)      # 恢复形状
#
#         x1 = x1.view(-1, 2704)
#         x = torch.cat([x1, x2], dim=1)
#         fc = self.feature_layer(x)
#         logits = self.classifier(fc)
#         probas = F.softmax(logits, dim=1)
#         return logits, probas, x1, x2, fc

# CNN-attention-Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba
# 自注意力机制模块（Self-Attention）
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, output_dim):
        super(MultiHeadSelfAttention, self).__init__()
        assert output_dim % num_heads == 0, "Output dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.fc_out = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V)

        weighted_values = weighted_values.transpose(1, 2).contiguous().view(B, L, -1)
        return self.fc_out(weighted_values)
#
# # 增强版的ConvNet模型
# class ConvNetWithMambaAndSelfAttention(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(ConvNetWithMambaAndSelfAttention, self).__init__()
#
#         # 2D-CNN 增加更多的卷积层（多层CNN）
#         self.net1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=5),    # 64-5+1=60 -> 60*60*16
#             nn.MaxPool2d(kernel_size=2),       # 60/2=30 -> 30*30*16
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, kernel_size=5),  # 30-5+1=26 -> 26*26*32
#             nn.MaxPool2d(kernel_size=2),       # 26/2=13 -> 13*13*32
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, kernel_size=3),  # 13-3+1=11 -> 11*11*64
#             nn.MaxPool2d(kernel_size=2),       # 11/2=5 -> 5*5*64
#             nn.ReLU(),
#             nn.BatchNorm2d(64)
#         )
#
#         # 1D-CNN 增加自注意力机制
#         self.net3 = nn.Sequential(
#             nn.Conv1d(1, 6, kernel_size=5),    # 433-5+1=429 -> 1*429*6
#             nn.MaxPool1d(kernel_size=3),       # 429/3=143 -> 1*143*6
#             nn.ReLU(),
#             nn.BatchNorm1d(6),
#             nn.Conv1d(6, 16, kernel_size=3),   # 143-3+1=141 -> 1*141*16
#             nn.MaxPool1d(kernel_size=3),       # 141/3=47 -> 1*47*16
#             nn.ReLU(),
#             nn.BatchNorm1d(16)
#         )
#
#         # 自注意力模块
#         self.attention = SelfAttention(input_dim=16, output_dim=16)
#
#         # 引入 Mamba 模块
#         self.mamba = Mamba(seq_len=47, d_model=16, state_size=128, device='cuda')
#
#         self.feature_layer = nn.Sequential(
#             nn.Linear(1600 + 752, 120),       # 全连接层（1600 + 752）
#             nn.ReLU(),
#             nn.Linear(120, 84),
#         )
#         self.classifier = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(84, num_classes),
#         )
#
#     def forward(self, x1, x2):
#         # x1 输入图像数据，x2 输入FFT信号数据
#         x1 = self.net1(x1)  # 提取图片特征
# #         print("x1 shape after CNN:", x1.shape)  # 打印出 x1 的形状
#         x2 = self.net3(x2)  # 提取FFT特征
# #         print("x2 shape after CNN:", x2.shape)  # 打印出 x2 的形状
#
#         # 自注意力机制：对1D特征应用注意力
#         x2 = x2.permute(0, 2, 1)   # 调整维度以匹配 Self-Attention 输入 (B, L, D)
#         x2 = self.attention(x2)    # 经过自注意力处理
#
#         # 使用 Mamba 处理 FFT 特征
#         x2 = self.mamba(x2)   # Mamba模块
#         x2 = x2.view(-1, 752)  # 恢复形状
#
#         # 2D图像特征变平（Flatten）
#         x1 = x1.view(-1, 1600)  # 拉平成(batch_size, 1600)
#
#         # 特征拼接
#         x = torch.cat([x1, x2], dim=1)  # 拼接成(batch_size, 1600 + 752)
#
#         # 全连接层
#         fc = self.feature_layer(x)
#         logits = self.classifier(fc)
#         probas = F.softmax(logits, dim=1)
#
#         return logits, probas, x1, x2, fc

# 改进版本的CNN-Attention-Mamba
# 1d卷积特征提取后加入注意力和Mamba增强
# 2d卷积特征提取后加入注意力增强
class ConvNetWithMambaAndSelfAttention(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNetWithMambaAndSelfAttention, self).__init__()

        # 2D-CNN（多层卷积+注意力机制）
        self.net1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),    # 64-5+1=60 -> 60*60*16
            nn.MaxPool2d(kernel_size=2),       # 60/2=30 -> 30*30*16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5),  # 30-5+1=26 -> 26*26*32
            nn.MaxPool2d(kernel_size=2),       # 26/2=13 -> 13*13*32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3),  # 13-3+1=11 -> 11*11*64
            nn.MaxPool2d(kernel_size=2),       # 11/2=5 -> 5*5*64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3),  # 5-3+1=3 -> 3*3*128
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # 这里调整为正确的输入输出维度
        self.attention2d = nn.Sequential(
            nn.Linear(128*3*3, 64),  # 输入维度是 128*3*3 (从卷积层的输出大小)，输出维度调整为64
            nn.ReLU(),
            nn.Linear(64, 128*3*3)    # 需要与前一层输出维度匹配
        )

        # 1D-CNN（多层卷积+注意力机制）
        self.net3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),    # 433-5+1=429 -> 1*429*16
            nn.MaxPool1d(kernel_size=3),       # 429/3=143 -> 1*143*16
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3),  # 143-3+1=141 -> 1*141*32
            nn.MaxPool1d(kernel_size=3),       # 141/3=47 -> 1*47*32
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3),  # 47-3+1=45 -> 1*45*64
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        self.attention1d = nn.Sequential(
            nn.Linear(64, 32),  # 自注意力头
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Mamba 模块
        self.mamba = Mamba(seq_len=45, d_model=64, state_size=128, device='cuda')

        # 特征融合与分类
        self.feature_layer = nn.Sequential(
            nn.Linear(128*3*3 + 64*45, 256),  # 特征维度调整
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),
        )

    def forward(self, x1, x2):
        # x1: 图像数据 (batch_size, 3, 64, 64)
        # x2: FFT信号 (batch_size, 1, signal_length)

        # 2D-CNN特征提取
        x1 = self.net1(x1)  # (batch_size, 128, 3, 3)
        x1 = x1.view(x1.size(0), -1)  # Flatten (batch_size, 128*3*3)
        x1 = self.attention2d(x1)  # 自注意力增强 (batch_size, 128*3*3)

        # 1D-CNN特征提取
        x2 = self.net3(x2)  # (batch_size, 64, 45)
        x2 = x2.permute(0, 2, 1)  # 调整维度 (batch_size, 45, 64)
        x2 = self.attention1d(x2)  # 自注意力增强 (batch_size, 45, 64)
        x2 = self.mamba(x2)  # Mamba模块 (batch_size, 128)
        x2 = x2.view(x2.size(0), -1)  # Flatten (batch_size, 128)

        # 特征融合
        x = torch.cat([x1, x2], dim=1)  # 拼接 (batch_size, 128*3*3 + 64*45)

        x = self.feature_layer(x)  # (batch_size, 256) -> (batch_size, 128) -> (batch_size, 64)

        # 分类
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas, x1, x2, x


model = ConvNetWithMambaAndSelfAttention(num_classes).to(device)
model = model.to(device)#传进device

from torchinfo import summary
# 多输入模型摘要
summary(model)
# 优化的学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # 初始学习率设为 1e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失

# 更新的训练代码
def compute_accuracy(model, feature, feature1, labels, batch_size):
    correct_pred, num_examples = 0, 0
    total_loss = 0
    N = feature.size(0)
    total_batch = int(np.ceil(N / batch_size))
    indices = np.arange(N)
    np.random.shuffle(indices)

    for i in range(total_batch):
        rand_index = indices[batch_size*i:batch_size*(i+1)]
        features = feature[rand_index,:]
        features1 = feature1[rand_index,:]
        targets = labels[rand_index]

        features = features.to(device)
        features1 = features1.to(device)
        targets = targets.to(device)

        logits, probas, _, _, _  = model(features, features1)
        cost = loss_fn(logits, targets)

        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        total_loss += cost.item()
        correct_pred += (predicted_labels == targets).sum()

    avg_loss = total_loss / num_examples
    avg_acc = (correct_pred.float() / num_examples) * 100
    return avg_loss, avg_acc

# 训练过程
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

best_valid_acc = 0.0  # 保存最佳验证精度

total_batch = int(np.ceil(N / batch_size))

for epoch in range(num_epochs):
    model.train()  # 训练模式（启用 Dropout 和 BN）
    running_loss = 0.0

    indices = np.arange(N)
    np.random.shuffle(indices)

    for i in range(0, N, batch_size):
        # 获取当前 batch 数据
        rand_index = indices[i:i + batch_size]
        features = train_features[rand_index, :].to(device)
        features2 = train_features2[rand_index, :].to(device)
        targets = train_labels[rand_index].to(device)

        # 前向传播
        logits, probas, _, _, _ = model(features, features2)
        cost = loss_fn(logits, targets)

        # 反向传播与参数更新
        optimizer.zero_grad()
        cost.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 增加最大梯度裁剪
        optimizer.step()

        running_loss += cost.item()

    # 计算训练集准确率
    train_loss_epoch, train_acc_epoch = compute_accuracy(model, train_features, train_features2, train_labels, batch_size)
    train_loss.append(train_loss_epoch)
    train_acc.append(train_acc_epoch)

    # 计算验证集准确率
    model.eval()  # 评估模式（关闭 Dropout 和 BN）
    with torch.no_grad():
        valid_loss_epoch, valid_acc_epoch = compute_accuracy(model, valid_features, valid_features2, valid_labels, batch_size)
        valid_loss.append(valid_loss_epoch)
        valid_acc.append(valid_acc_epoch)

    # 打印当前epoch的训练和验证精度及损失
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss_epoch:.4f}, Train Accuracy: {train_acc_epoch:.2f}%, "
          f"Valid Loss: {valid_loss_epoch:.4f}, Valid Accuracy: {valid_acc_epoch:.2f}%")

    # 更新学习率
    scheduler.step(running_loss)

    # 保存最佳模型
    if valid_acc_epoch > best_valid_acc:
        best_valid_acc = valid_acc_epoch
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")

# 输出最终结果
print(f"Best Validation Accuracy: {best_valid_acc:.2f}%")

# 训练结束后保存最终模型
torch.save(model, 'model/Conv-mmamba_CNN33.pkl')

# 确保数据转换为 numpy 数组
plt.figure()
train_loss_np = np.array(train_loss)
valid_loss_np = np.array(valid_loss)
plt.plot(train_loss_np, label='train')
plt.plot(valid_loss_np, label='valid')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Accuracy 曲线
plt.figure()
train_acc_np = np.array([acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in train_acc])
valid_acc_np = np.array([acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in valid_acc])
plt.plot(train_acc_np, label='train')
plt.plot(valid_acc_np, label='valid')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# In[]
# In[6]: 利用训练好的模型 对测试集进行分类

model=torch.load('model/Conv-mmamba_CNN33.pkl')#加载模型
#提取测试集图片
x_test,y_test=read_directory('小波时频33/test_img',height,width,normal=1)
x_test2=datafft['test_X']

x_test2=ss2.transform(x_test2)

x_test2=x_test2.reshape(x_test2.shape[0],1,-1)
test_features = torch.tensor(x_test).type(torch.FloatTensor)
test_features2 = torch.tensor(x_test2).type(torch.FloatTensor)
test_labels = torch.tensor(y_test-1).type(torch.LongTensor)
# test_labels = torch.tensor(y_test).type(torch.LongTensor)

model = model.eval()
_, teac = compute_accuracy(model, test_features, test_features2, test_labels, batch_size)
print('测试集正确率为：',teac.item(),'%')

# In[] 可视化
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#
# # 确保模型已加载到设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 验证集预测
# with torch.no_grad():
#     valid_features = torch.tensor(x_valid).to(device).float()  # 转换为张量
#     valid_features2 = torch.tensor(x_valid2).to(device).float()
#     _, _, _, _, fc_valid_features = model(valid_features, valid_features2)
#
#     # Softmax 概率并生成预测值
#     probabilities_valid = torch.softmax(fc_valid_features, dim=1)
#     predictions = torch.argmax(probabilities_valid, dim=1).cpu().numpy()  # 转为 NumPy
#
# # 调整真实标签（如从 1 开始，则需要减去 1）
# y_valid = np.array(y_valid)  # 确保标签是 NumPy 数组
# if y_valid.min() == 1:  # 如果标签范围是 [1, num_classes]
#     y_valid -= 1
#
# # 检查长度是否一致
# if len(predictions) != len(y_valid):
#     raise ValueError(f"Inconsistent lengths: predictions({len(predictions)}) and y_valid({len(y_valid)})")
#
# # 生成混淆矩阵
# num_classes = len(np.unique(y_valid))  # 确定分类数量
# conf_matrix = confusion_matrix(y_valid, predictions, labels=range(num_classes))
#
# # 可视化混淆矩阵
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(num_classes))
# disp.plot(cmap="Blues", values_format="d")
# plt.title("Confusion Matrix for Validation Data")
# plt.show()
from sklearn.manifold import TSNE
# 自动生成颜色映射
colors = ['blue', 'purple', 'yellow','magenta', 'red', 'lime', 'cyan', 'orange']
# colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
# colors = ['black', 'blue', 'purple', 'yellow', 'magenta', 'red', 'lime', 'cyan', 'orange', 'gray']
# colors = ['black', 'blue', 'purple', 'yellow', 'magenta', 'red', 'lime']
# colors = ['blue', 'purple', 'red']
tsne = TSNE(n_components=2)
# 原始数据 降维
X1=tsne.fit_transform(loadmat('data_process33.mat')['test_X'])
X1=(X1-X1.min(axis=0))/(X1.max(axis=0)-X1.min(axis=0))
Y=loadmat('data_process33.mat')['test_Y'].argmax(axis=1)
Y = Y - Y.min()  # 确保标签从 0 开始
plt.figure()
for i in range(num_classes):
# for i in range(5):
#     plt.plot(X1[Y==i,0],X1[Y==i,1],'*',markersize=5,label=str(i),c=colors[i])
   plt.plot(X1[i,0],X1[i,1],'*',color=plt.cm.Set1(Y[i]),markersize=5,label=str(Y[i]))

plt.legend()
plt.xlabel('first component')
plt.ylabel('second component')
plt.title('input_data')
plt.show()

in_1 = test_features.to(device)
in_2 = test_features2.to(device)
_, _,_,_,fc_feature = model(in_1,in_2)

# 将数据输入到训练好的双流cnn中，提取最后一个全连接层的输出
fc_feature = fc_feature.to('cpu').detach().numpy()  # 确保正确转换
# 将输入数据与全连接层输出都降到2维，实现可视化

# 特征数据 降维
X2=tsne.fit_transform(fc_feature)
X2=(X2-X2.min(axis=0))/(X2.max(axis=0)-X2.min(axis=0))
Y=y_test

plt.figure()
# for i in range(7):
#     plt.plot(X2[Y==i,0],X2[Y==i,1],'*',markersize=5,label=str(i),c=colors[i])
for i in range(num_classes):
    plt.plot(X2[Y==i,0],X2[Y==i,1],'*',markersize=5, c=colors[i])
plt.legend()
plt.xlabel('first component')
plt.ylabel('second component')
plt.title('feature_data')
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

# colors = ['black', 'blue', 'purple', 'yellow', 'magenta', 'red', 'lime', 'cyan', 'orange', 'gray']
# colors = ['black', 'blue', 'purple', 'yellow', 'magenta', 'red', 'lime', 'cyan']
# colors = ['blue', 'purple', 'red']
# 进行 t-SNE 降维到 3 维
tsne = TSNE(n_components=3)
X1 = tsne.fit_transform(loadmat('data_process33.mat')['test_X'])

# 归一化数据
X1 = (X1 - X1.min(axis=0)) / (X1.max(axis=0) - X1.min(axis=0))

# 获取标签
Y = loadmat('data_process33.mat')['test_Y'].argmax(axis=1)
# 创建三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制每个类的点，使用不同的颜色
for i in range(num_classes):
#     ax.scatter(X1[Y == i, 0], X1[Y == i, 1], X1[Y == i, 2],
#                marker='*', s=50,label=str(i), c=colors[i])
# for i in range(5):
    ax.scatter(X1[Y == i, 0], X1[Y == i, 1], X1[Y == i, 2],
               marker='*', s=50, color=colors[i])
# 设置标签
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.set_zlabel('Third component')
ax.set_title('t-SNE 3D visualization of input_data')
# 添加图例
ax.legend()
# 显示图形
plt.show()

from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图库
# 特征数据降维到 3 维
tsne = TSNE(n_components=3)  # 设置为 3 维
X2 = tsne.fit_transform(fc_feature)
X2 = (X2 - X2.min(axis=0)) / (X2.max(axis=0) - X2.min(axis=0))
Y = y_test  # 保持标签不变

# 绘制 3D t-SNE 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 创建 3D 图
# for i in range(5):  # 适用于 10 个类别
#     ax.scatter(X2[Y == i, 0], X2[Y == i, 1], X2[Y == i, 2],
#                marker='*', s=20,label=str(i), c=colors[i])
for i in range(num_classes):  # 适用于 10 个类别
    ax.scatter(X2[Y == i, 0], X2[Y == i, 1], X2[Y == i, 2],
               marker='*', s=20, color=colors[i])
ax.legend()
ax.set_xlabel('First Component')
ax.set_ylabel('Second Component')
ax.set_zlabel('Third Component')
ax.set_title('3D Feature Visualization')
plt.show()

# Extract features from the validation set
in_valid1 = valid_features.to(device)
in_valid2 = valid_features2.to(device)

# Ensure model is in evaluation mode
model.eval()
_, _, _, _, fc_valid_features = model(in_valid1, in_valid2)

# Move to CPU and detach for numpy conversion
fc_valid_features = fc_valid_features.to('cpu').detach().numpy()

# Perform t-SNE for 3D reduction
tsne = TSNE(n_components=3)
X_valid_3d = tsne.fit_transform(fc_valid_features)
X_valid_3d = (X_valid_3d - X_valid_3d.min(axis=0)) / (X_valid_3d.max(axis=0) - X_valid_3d.min(axis=0))

# Get the validation labels for plotting
Y_valid = y_valid  # Assuming `y_valid` is already loaded as a numpy array

# Plot the 3D t-SNE visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# for i in range(num_classes):
#     ax.scatter(X_valid_3d[Y_valid == i, 0], X_valid_3d[Y_valid == i, 1], X_valid_3d[Y_valid == i, 2],
#                marker='*', s=20, label=str(i),c=colors[i])
# Plot each class with a different color
for i in range(num_classes):
    ax.scatter(X_valid_3d[Y == i, 0], X_valid_3d[Y == i, 1], X_valid_3d[Y == i, 2],
                   marker='*', s=20, color=colors[i])

# Add plot labels and title
ax.set_xlabel('First Component')
ax.set_ylabel('Second Component')
ax.set_zlabel('Third Component')
ax.set_title('3D t-SNE Visualization of Validation Features')

# Add legend and show plot
ax.legend()
plt.show()
from torchinfo import summary
# 多输入模型摘要
summary(model, input_size=[(1, 3, 64, 64), (1, 1, x_train2.shape[-1])], device='cpu')



























