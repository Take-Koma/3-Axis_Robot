
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# --- 設定（学習時と合わせる） ---
MODEL_PATH = "WorkpieceCenterModel.onnx"
TEST_IMAGE = "./train_images/work_622_91.jpg" # 確認したい画像パス
INPUT_SIZE = 224

# 1. ONNXモデルの読み込み
# T600(GPU)を使う設定。もしエラーが出るなら ['CPUExecutionProvider'] に変更
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])

# 2. 画像の読み込みと前処理
# 学習時と同じ「リサイズ」と「正規化(0-1)」を行う
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])

img = Image.open(TEST_IMAGE).convert('RGB')
input_tensor = transform(img).unsqueeze(0).numpy() # (1, 3, 224, 224)の形に

# 3. 推論実行
inputs = {session.get_inputs()[0].name: input_tensor}
outputs = session.run(None, inputs)

# 4. 結果の解析
# 出力は [u_normalized, v_normalized] の形
pred_u_norm, pred_v_norm = outputs[0][0]

# ピクセル座標に復元 (カメラ解像度 816x624 を掛ける)
pred_u = pred_u_norm * 816.0
pred_v = pred_v_norm * 624.0

print(f"--- 推論結果 ---")
print(f"予測座標 (U, V): {pred_u:.2f}, {pred_v:.2f}")
print(f"元ファイル名の座標と比較して、近い値が出ていれば成功です！")