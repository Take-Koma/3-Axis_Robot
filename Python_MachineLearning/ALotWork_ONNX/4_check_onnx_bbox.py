
import onnxruntime as ort
import numpy as np
from PIL import Image

# --- 設定（今回のONNXの仕様に合わせる） ---
MODEL_PATH = "WorkDetectTopK.onnx"
TEST_IMAGE = "dataset_raw/images/work_525_594.jpg" # テストしたい画像（どれでもOK）
INPUT_SIZE = 640  # 今回のONNXの約束サイズ

def main():
    print("【テスト開始】ONNXモデルの推論テストを行います。")

    # 1. ONNXモデルの読み込み（T600 GPUを使用、なければCPU）
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"使用中のデバイス: {session.get_providers()}")

    # 2. 画像の読み込み
    img_org = Image.open(TEST_IMAGE).convert('RGB')
    org_w, org_h = img_org.size # 元のサイズ（816, 624）を記憶

    # 3. 前処理（アスペクト比を無視して 640x640 にギュッとリサイズ）
    img_resized = img_org.resize((INPUT_SIZE, INPUT_SIZE))
    
    # ピクセル値を 0〜255 から 0.0〜1.0 の NumPy配列に変換
    input_data = np.array(img_resized, dtype=np.float32) / 255.0
    # 画像の並びを [縦, 横, チャンネル] から [チャンネル, 縦, 横] に並び替える
    input_data = np.transpose(input_data, (2, 0, 1))
    # 先頭にバッチ用の次元を追加して (1, 3, 640, 640) の形にする
    input_tensor = np.expand_dims(input_data, axis=0)

    # 4. 推論実行
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)

    # ONNXの出力（50個固定配列）をそれぞれ受け取る
    # outputs[0]: boxes_xyxy (50, 4)
    # outputs[1]: scores     (50,)
    # outputs[2]: labels     (50,)
    boxes_xyxy = outputs[0]
    scores = outputs[1]
    labels = outputs[2]

    # --- 🔽 ここから追加 🔽 ---
    # もし1番目の検出スコアが完全に0なら、1個も見つからなかったと判定して安全に終了する
    if scores[0] == 0:
        print("\n【確認】ONNXモデルは正常に動いていますが、この画像からはワークが1個も見つかりませんでした。")
        return
    # --- 🔼 ここまで追加 🔼 ---

    # 5. 結果の解析（先ほどのTwinCAT用の計算フローと同じ動きをPythonで再現！）
    print(f"\n--- 推論結果（自信度 70% 以上のワークを表示） ---")
    threshold = 0.7
    detect_count = 0

    for i in range(50):
        score = scores[i]
        # 自信度が閾値を超えているものだけを表示
        if score >= threshold:
            detect_count += 1
            # 640x640 の世界での四角の座標 [左上X, 左上Y, 右下X, 右下Y]
            x1, y1, x2, y2 = boxes_xyxy[i]

            # 【重要】元のカメラ解像度（816x624）に換算（復元）する
            real_x1 = x1 * (org_w / float(INPUT_SIZE))
            real_y1 = y1 * (org_h / float(INPUT_SIZE))
            real_x2 = x2 * (org_w / float(INPUT_SIZE))
            real_y2 = y2 * (org_h / float(INPUT_SIZE))

            # ワークの「中心点」を計算
            center_x = (real_x1 + real_x2) / 2.0
            center_y = (real_y1 + real_y2) / 2.0

            print(f"ワーク [{detect_count}]: 自信度={score*100:.1f}%")
            print(f"  └─ 予測中心ピクセル (X, Y): {center_x:.2f}, {center_y:.2f}")
            print(f"  └─ 検出枠 [左上X, 左上Y, 右下X, 右下Y]: [{real_x1:.1f}, {real_y1:.1f}, {real_x2:.1f}, {real_y2:.1f}]")
        else:
            # 自信度が高い順に並んでいるので、下回ったらそこでループ終了
            break

    if detect_count == 0:
        print("【警告】ワークが見つかりませんでした。閾値を下げるか、画像パスを確認してください。")
    else:
        print(f"\n合計 {detect_count} 個のワークを検出しました！")
        print("ファイル名にある座標（例: work_113_588 なら X=113, Y=588）と近い値が出ていれば大成功です！")

if __name__ == "__main__":
    main()