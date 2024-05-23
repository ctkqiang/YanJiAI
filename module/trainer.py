# -*- coding: UTF-8 -*-
try:
    import os
    import torch
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import onnx
    from datetime import datetime
    from onnx2keras import onnx_to_keras
    from tensorflow.keras.models import save_model
    from PIL import Image
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split
except ImportError:
    raise ImportError("🥹无法安装配件")
finally:
    pass

class 简单卷积神经网络(nn.Module):
    def __init__(self):
        super(简单卷积神经网络, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class 训练器:
    def __init__(self, 文件名称: str) -> None:
        super(训练器, self).__init__()

        self.文件名称: str = 文件名称
        self.fill_mode: str = "nearest"
        self.batch_size: int = 32  
        self.model = 简单卷积神经网络()

        assert torch.__version__.startswith("1")

    

    def 检查图片(self, 目录: str) -> None:
        for filename in os.listdir(目录):
            
            if filename.endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(os.path.join(目录, filename))
                    img.verify()
                except (IOError, SyntaxError) as e:
                    print("\033[0;93m" + f"坏文件: {filename}")  

    def 准备数据(self, 训练数据: str, 测试数据: str):
        if not os.path.exists(训练数据):
            raise FileNotFoundError(f"训练数据目录 '{训练数据}' 不存在.")
        
        if not os.path.exists(测试数据):
            raise FileNotFoundError(f"测试数据目录 '{测试数据}' 不存在.")

        self.检查图片(训练数据)
        self.检查图片(测试数据)

        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])

        dataset = ImageFolder(root=训练数据, transform=transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def 训练模型(self, train_loader, val_loader, 轮数=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(轮数):
            
            self.model.train()

            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)

            print("\033[0;93m" + f"轮 {epoch + 1}/{轮数}, 损失: {epoch_loss:.4f}")

    def 保存pytorch模型(self, 路径="model.pth"):
        torch.save(self.model.state_dict(), 路径)

    def 打印onnx节点名称(self, onnx_model_path):
        model = onnx.load(onnx_model_path)

        for node in model.graph.node:
            print("\033[0;93m" + f'节点名称: {node.name}')
            
        for input in model.graph.input:
            print("\033[0;93m" + f'输入名称: {input.name}')
            
        for output in model.graph.output:
            print("\033[0;93m" + f'输出名称: {output.name}')


    def 转换为keras(self, pytorch_model_path, keras_model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = 简单卷积神经网络()
        model.load_state_dict(torch.load(pytorch_model_path))
        model.eval()
        
        dummy_input = torch.randn(1, 3, 150, 150, device=device)
        onnx_path = f"{self.文件名称}.onx"
        torch.onnx.export(model, dummy_input, onnx_path)


        self.打印onnx节点名称(onnx_path)
        
        onnx_model = onnx.load(onnx_path)

        # k_model = onnx_to_keras(onnx_model, ["input"]) 
        # save_model(k_model, keras_model_path)

    def 运行(self, 训练数据: str, 测试数据: str) -> None:
        train_loader, val_loader = self.准备数据(训练数据, 测试数据)
        self.训练模型(train_loader, val_loader)
        self.保存pytorch模型(f"{self.文件名称}.pth")
        self.转换为keras(f"{self.文件名称}.pth", "model.h5")

    def 转换_pth_到_h5(pth_模型路径, h5_模型路径):
        # 加载 PyTorch 模型
        模型 = torch.load(pth_模型路径)
        模型.eval()

        # 用于 ONNX 导出的虚拟输入
        虚拟输入 = torch.randn(1, 3, 150, 150)

        # 导出模型到 ONNX
        onnx_模型路径 = pth_模型路径.replace(".pth", ".onnx")
        torch.onnx.export(模型, 虚拟输入, onnx_模型路径, input_names=["input"], output_names=["output"])

        # 加载 ONNX 模型
        onnx_模型 = onnx.load(onnx_模型路径)

        # 将 ONNX 模型转换为 Keras 模型
        k_模型 = onnx_to_keras(onnx_模型, ["input"])

        # 将 Keras 模型保存为 H5 文件
        save_model(k_模型, h5_模型路径)
