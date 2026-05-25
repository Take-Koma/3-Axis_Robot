import os
import subprocess
import sys

# このスクリプトがある場所（YOLOX_CALLフォルダ）のパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1つ上の階層に上がり、そこから「YOLOX」フォルダの絶対パスを作ります
yolox_dir = os.path.abspath(os.path.join(current_dir, "..", "YOLOX"))

# 作業ディレクトリを「YOLOX」フォルダに変更
os.chdir(yolox_dir)

# 実行したいコマンドをリスト形式で定義
cmd = [
    sys.executable,                                                         # 1. 実行中のPython環境
    "tools/export_onnx.py",                                                 # 2. ONNX変換用スクリプト
    "-f", "exps/default/yolox_nano_work.py",                                # 3. [-f] 使用したネットワーク構成ファイル
    "-c", "./YOLOX_outputs/yolox_nano_work/latest_ckpt.pth",                # 4. [-c] 変換元となる学習済みの重み（最新チェックポイント）
    "--output-name", "./YOLOX_outputs/yolox_nano_work/YOLOXmodel.onnx",     # 5. [--output-name] 出力されるONNXファイルの保存先
    "--decode_in_inference"                                                 # 6. [--decode_in_inference] ONNXモデル内で推論時のデコード処理も一緒に組み込むフラグ（これがあると、ONNXモデル単体で推論が完結するようになります）
]

print("=" * 60)
print(f"🚀 YOLOX Export ONNX Started via YOLOX_CALL")
print(f"Working Dir: {yolox_dir}")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

# コマンドを実行
subprocess.run(cmd)