
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ==================================================================================
# ① TwinCAT向けの軽量CNNモデルの定義
# ==================================================================================
class LightweightOCR(nn.Module):
    def __init__(self, num_classes=10):  # num_classes: 分類したい文字数（数字のみなら10）
        super(LightweightOCR, self).__init__()
        
        # 特徴抽出層（畳み込み層）
        self.features = nn.Sequential(
            # 入力: [1, 1, 28, 28] (Batch, Channel(白黒), Height, Width)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # サイズが 28x28 -> 14x14 に縮小
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # サイズが 14x14 -> 7x7 に縮小
        )
        
        # 分類層（全結合層）
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # 出力: 各文字（0〜9）である確率の元（ロジット）
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # バッチ次元を残して1次元に平坦化
        x = self.classifier(x)
        return x


def main():
    # GPUが使える環境（CUDA）ならGPUを使い、なければCPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ==================================================================================
    # ② データの準備（MNISTデータセットの読み込み）
    # ==================================================================================

    dataSet_Select = input("データセットを選択してください（1: MNIST, 2: 実データセット）: ")

    if dataSet_Select == "1":

        # -----MNISTデータセットの読み込み用のコード-----
        # 画像の変形ルール（PyTorchのテンソルに変換 ＆ 0.0〜1.0 に正規化）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 訓練用データとテスト用データのダウンロード・読み込み
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # DataLoaderを使うことで、データを指定したバッチサイズ（例: 64枚ずつ）に小分けにしてモデルに投入できます
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)
        # -----MNISTデータセットの読み込み用のコードここまで-----

    elif dataSet_Select == "2":

        # -----実データセットの読み込み用のコード-----
        # 白黒（1チャンネル）で読み込み、28x28にリサイズする変形ルール
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((28, 28)),

            # 👇 ここが水増しの魔法
            transforms.RandomRotation(degrees=10),               # 💡 ワークの傾き対策（±10度のランダム回転）
            transforms.RandomPerspective(distortion_scale=0.1), # 💡 カメラのあおり角対策（斜めからの歪み）
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # 💡 ピントボケ・油汚れ対策（ブラー）

            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 自分で用意したフォルダを指定するだけ！
        train_set = torchvision.datasets.ImageFolder(root='./dataset', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        # -----実データセットの読み込み用のコードここまで-----



    # ==================================================================================
    # ③ モデル、損失関数、最適化アルゴリズムの設定
    # ==================================================================================
    model = LightweightOCR(num_classes=10).to(device)
    
    # 損失関数（正解の文字と、モデルの予測のズレを計算する数式）
    criterion = nn.CrossEntropyLoss()
    
    # 最適化（ズレを減らすようにモデルの重みを賢く更新する職人）
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==================================================================================
    # ④ 学習ループ（model.train()）
    # ==================================================================================
    epochs = 3  # データの全周を何回繰り返すか（MNISTなら3回でも98%以上の精度が出ます）
    
    for epoch in range(epochs):
        model.train()  # 🚨モデルを「学習モード」に設定
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()       # 1. 過去の勾配（計算の残りカス）をクリア
            outputs = model(images)     # 2. モデルに画像を投入して予測（順伝播）
            loss = criterion(outputs, labels) # 3. 正解とのズレ（損失）を計算
            loss.backward()             # 4. ズレを元に、どのパラメータを直すべきか逆計算（逆伝播）
            optimizer.step()            # 5. パラメータを実際に更新
            
            running_loss += loss.item()
            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/200:.4f}")
                running_loss = 0.0

    print("学習が完了しました！")

    # ==================================================================================
    # ⑤ 簡易テスト（精度の確認）
    # ==================================================================================
    # 💡 修正箇所：MNIST（1）を選択したときだけテストを実行する（実データのときはスキップ）
    if dataSet_Select == "1":
        model.eval()  # 🚨モデルを「評価モード」に設定
        correct = 0
        total = 0
        with torch.no_grad(): # テスト時はメモリ節約のため勾配計算をオフにする
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print(f"テスト精度: {100 * correct / total:.2f} %")
    else:
        print("実データセットのため、簡易テストをスキップしてONNX出力へ進みます。")

    # ==================================================================================
    # ⑥ TwinCAT用 ONNXエクスポート
    # ==================================================================================
    model.eval()# 🚨エクスポート前には必ず推論モード（eval）にする
    # TwinCATはサイズ可変を許さないため、ここで入力サイズを完全に固定します。
    # [BatchSize=1, Channel=1, Height=28, Width=28]
    dummy_input = torch.randn(1, 1, 28, 28).to(device) # TwinCAT固定サイズ [1, 1, 28, 28]
    onnx_file_path = "factory_ocr_model.onnx"

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path, 
        export_params=True,                 # モデルの重み（学習済みのパラメータ）を一緒に保存する
        opset_version=12,                   # 🚨 TwinCATが安定してサポートしているバージョンを指定
        do_constant_folding=True,           # グラフの最適化（不要な計算を事前にまとめる）
        input_names=['input_image'],        # TwinCAT側で表示される入力ポート名
        output_names=['output_logits']      # TwinCAT側で表示される出力ポート名
        # 🚨 注意：Pythonでよく使う 'dynamic_axes'（可変サイズ指定）は絶対に書かない！
    )
    print(f"TwinCAT用ONNXモデルを出力しました: {onnx_file_path}")

if __name__ == "__main__":
    main()
