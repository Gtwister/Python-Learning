import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import pickle
import copy
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --- 1. 模型架构 (严格保持你的 LeNet 拓扑) ---
class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 0),  
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        self.out = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        self._init_weights() # 新增：权重初始化

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # 使用 Xavier 初始化提升收敛速度
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

# --- 2. 数据加载与增强预处理 ---
def load_and_preprocess(): 
    with gzip.open("./dataset/mnist/mnist.pkl.gz", 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    def resize_and_norm(images):
        resized = []
        for img in images:
            # Resize 28x28 -> 32x32
            pil_img = Image.fromarray(img.reshape(28, 28))
            resized_img = pil_img.resize((32, 32)) #（32,32）
            resized.append(np.array(resized_img))#(n , 32, 32)
        # 归一化至 0-1 之间 (解决 10% 准确率的关键)
        return torch.from_numpy(np.array(resized)).float().unsqueeze(1) / 255.0 # (n,1,32,32)

    return (resize_and_norm(x_train), torch.tensor(y_train, dtype=torch.long),
            resize_and_norm(x_valid), torch.tensor(y_valid, dtype=torch.long),
            resize_and_norm(x_test), torch.tensor(y_test, dtype=torch.long))

# 准备 DataLoader
x_train, y_train, x_valid, y_valid, x_test, y_test = load_and_preprocess()
train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
valid_dl = DataLoader(TensorDataset(x_valid, y_valid), batch_size=128)
test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=128)

dataloaders = {'train': train_dl, 'valid': valid_dl}

# --- 3. 训练函数 (加入早停和调度器修正) ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50, filename='LeNet-5_Best.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 5  # 连续5个epoch不提升则停止
    counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)# [[0.1,0.2,0.6,0.7,0.8,0.9,0.3,0.4,0.5,0.0],[]......]
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 只有在 valid 阶段更新调度器和保存模型
            if phase == 'valid':
                scheduler.step() # StepLR 不需要传参数
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, filename)
                    counter = 0
                else:
                    counter += 1

        if counter >= patience:
            print("Early stopping triggered!")
            break
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}\n")

    model.load_state_dict(best_model_wts)
    return model

# --- 4. 执行训练与最终验证 ---
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam 通常比 SGD 更快更稳
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # 每10步降一半

# 开始训练
model_ft = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=30)

# 最终在测试集上展示结果

# 2. 准备测试数据加载器
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 改进后的测试与可视化函数
def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(16, 8))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        img = images[i].cpu().squeeze()
        plt.imshow(img, cmap='gray')
        
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 执行可视化
visualize_predictions(model_ft, test_dataloader, device)

