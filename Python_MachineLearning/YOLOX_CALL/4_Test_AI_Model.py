import os
import random  # 💡 追加：リストからランダムに要素を選ぶためのライブラリ
import glob    # 💡 追加：フォルダ内のファイル一覧を検索して取得するためのライブラリ
import subprocess
import sys

# このスクリプトがある場所（YOLOX_CALLフォルダ）のパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1つ上の階層に上がり、そこから「YOLOX」フォルダの絶対パスを作ります
yolox_dir = os.path.abspath(os.path.join(current_dir, "..", "YOLOX"))

# 作業ディレクトリを「YOLOX」フォルダに変更
os.chdir(yolox_dir)

# 💡 【ここが新しい処理！】
# 1. 画像が入っているフォルダのパスを指定
image_folder = "./datasets/VOCdevkit/VOC2007/JPEGImages"

# 2. フォルダ内にある「.jpg」で終わるファイルのパスを一斉に取得してリスト（配列）にします
#（Excelマクロでいう「Dir関数」でワイルドカードを使ってファイルを全部集めるイメージです）
jpg_files = glob.glob(os.path.join(image_folder, "*.jpg"))

# 3. 集めたファイルリストから、ランダムに1つだけ選ぶ
if jpg_files:
    selected_image = random.choice(jpg_files)  # リストの中から神の御心で1つチョイス
else:
    # 万が一、フォルダが空っぽだった場合の安全装置（フォールバック）
    selected_image = "./datasets/VOCdevkit/VOC2007/JPEGImages/1.jpg"
    print("⚠️ JPGファイルが見つからなかったため、デフォルトの1.jpgを使用します。")


# 実行したいコマンドをリスト形式で定義
cmd = [
    sys.executable,                                             # 1. 実行中のPython
    "tools/demo.py",                                            # 2. YOLOXの推論スクリプト
    "image",                                                    # 3. モード指定：「静止画」
    "-f", "exps/default/yolox_nano_work.py",                    # 4. [-f] 使用するネットワーク構成ファイル
    "-c", "./YOLOX_outputs/yolox_nano_work/latest_ckpt.pth",    # 5. [-c] 最新の重み
    "--path", selected_image,                                   # 6. [--path] 💡 上でランダムに選ばれた画像のパスが入ります！
    "--conf", "0.25",                                           # 7. [--conf] 検出閾値
    "--nms", "0.45",                                            # 8. [--nms] 重複した枠を一本化する感度
    "--tsize", "640",                                           # 9. [--tsize] 画像の処理サイズ
    "--device", "cpu",                                          # 10. [--device] 推論を「CPU」で実行
    "--save_result"                                             # 11. [--save_result] 検出結果を描き込んだ画像を保存
]

print("=" * 60)
print(f"🚀 YOLOX Test AI Model (Random Selection) Started via YOLOX_CALL")
print(f"Working Dir: {yolox_dir}")
print(f"Selected Image: {selected_image}")  # 💡 画面にどの画像が選ばれたか表示する親切設計
print("=" * 60)

# コマンドを実行
subprocess.run(cmd)