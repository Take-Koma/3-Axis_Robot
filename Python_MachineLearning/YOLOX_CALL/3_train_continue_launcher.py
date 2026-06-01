import os
import shutil
import subprocess
import sys

# ==============================================================================
# 🚀 YOLOX 学習実行処理
# ====================================================================

# 1つ上の階層に上がり、そこから「YOLOX」フォルダの絶対パスを作ります
yolox_dir = os.path.abspath(os.path.join(current_dir, "..", "YOLOX"))

# 作業ディレクトリを「YOLOX」フォルダに変更（これで ./YOLOX_outputs がそのまま通ります）
os.chdir(yolox_dir)

# 実行したいコマンドをリスト形式で定義
cmd = [
    sys.executable, 
    "tools/train.py",
    "-f", "exps/default/yolox_nano_work.py",
    "-d", "1",
    "-b", "4",
    "--fp16",
    "-c", "./YOLOX_outputs/yolox_nano_work/latest_ckpt.pth",    # 【ここが2_train_launcher.pyとの変更点！】前回の学習で出力された最新の重み（AIの脳みそ）を指定
    "-o"
]

print("=" * 60)
print(f"🚀 YOLOX Continue Training Started via YOLOX_CALL")
print(f"Working Dir: {yolox_dir}")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

# コマンドを実行（確認だけにしたいときは、頭に # をつけてコメントアウトしてください）
subprocess.run(cmd)