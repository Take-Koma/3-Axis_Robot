# -*- coding: utf-8 -*-
"""
Robot Workpiece Detection - Training & ONNX Export Script
Target: NVIDIA T600 GPU & TwinCAT Vision (TF7810)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import glob

# ==========================================
# 1. 各種設定
# ==========================================
IMAGE_DIR = './train_images'      # 学習用画像フォルダ
MODEL_NAME = "WorkpieceCenterModel.onnx"
INPUT_SIZE = 224                  # AIモデルの入力サイズ (224x224)
BATCH_SIZE = 4                    # 一度に学習する画像数 (T600なら4〜8程度)
NUM_EPOCHS = 50                   # 学習の繰り返し回数
LEARNING_RATE = 0.001             # 学習率

# ==========================================
# 2. データセット定義（画像と座標の紐付け）
# ==========================================
class CenterDataset(Dataset):
    """
    ファイル名 "work_U_V.jpg" から座標を取得するカスタムデータセット
    """
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path).convert('RGB')
        
        # ファイル名（例: work_550_269）を分解して座標を取得
        filename = os.path.basename(path).replace('.jpg', '')
        try:
            _, u, v = filename.split('_')
        except ValueError:
            raise ValueError(f"ファイル名の形式が不正です: {filename} (期待値: name_U_V.jpg)")
        
        # 座標の正規化 (0.0〜1.0)
        # 座標の正規化 (816x624に合わせて調整)
        target = torch.tensor([float(u)/816.0, float(v)/624.0], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        return image, target

# ==========================================
# 3. 学習の準備
# ==========================================
# デバイスの設定（NVIDIA T600 を使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# データ変換の設定 (リサイズ ＆ テンソル化)
transform_pipeline = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])

# データセットの実体化
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"警告: {IMAGE_DIR} フォルダが見つからなかったため作成しました。画像を配置してください。")

dataset = CenterDataset(img_dir=IMAGE_DIR, transform=transform_pipeline)

# AIモデルの定義 (軽量・高速な MobileNetV2)
model = models.mobilenet_v2(pretrained=True)
# 出力層を「座標2個(u, v)」に書き換え
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.to(device)

# 最適化手法と誤差関数の設定
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()  # 座標予測（回帰）に最適な二乗誤差

# ==========================================
# 4. 学習実行 (Training Loop)
# ==========================================
if len(dataset) > 0:
    print(f"\n--- 学習開始 (Total images: {len(dataset)}) ---")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 10回ごとに進捗を表示
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(dataset):.6f}")

    print("学習完了！")
else:
    print("\n警告: 学習用画像がないため、未学習の状態でONNXを出力します。")

# ==========================================
# 5. ONNXエクスポート
# ==========================================
model.eval()
print(f"\n--- ONNXエクスポート開始 ---")

# ダミーデータ（モデルの入力をシミュレート）
dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

torch.onnx.export(
    model,
    dummy_input,
    MODEL_NAME,
    export_params=True,
    opset_version=11,      # TwinCAT Vision (TF7810) 互換バージョン
    input_names=['input'],
    output_names=['output'],
    # バッチサイズを可変にする設定（TwinCAT側の柔軟性のため）
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"成功: '{MODEL_NAME}' を出力しました。")
print(f"このファイルを TwinCAT プロジェクトのしかるべきフォルダへ配置してください。")