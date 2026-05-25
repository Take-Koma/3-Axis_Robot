import os
import shutil
import subprocess
import sys

# ==============================================================================
# 🔄 【自動化】Cドライブ側にデータセットがなければ、OneDriveから自動コピーする処理
# ==============================================================================
# 1. コピー先のパス（Cドライブ側の引っ越し先）
dst_voc2007 = r"C:\Python_Venvs\datasets\VOCdevkit\VOC2007"

# 2. コピー元のパス（このスクリプトがあるYOLOX_CALLから見たOneDrive側の元の場所を自動計算）
current_dir = os.path.dirname(os.path.abspath(__file__))
src_voc2007 = os.path.abspath(os.path.join(current_dir, "..", "YOLOX", "datasets", "VOCdevkit", "VOC2007"))

print("=" * 60)
print("🔍 データセットの配置チェック...")

if not os.path.exists(dst_voc2007):
    print(f"🚨 Cドライブ側に VOC2007 フォルダが見つかりません。")
    print(f"📦 OneDrive（Gitリポジトリ側）から自動コピーを開始します...")
    print(f"   [元] {src_voc2007}")
    print(f"   [先] {dst_voc2007}")
    
    try:
        # コピー先の親フォルダ（C:\Python_Venvs\datasets\VOCdevkit）がなければ自動作成
        os.makedirs(os.path.dirname(dst_voc2007), exist_ok=True)
        
        # フォルダを中身（Annotations, JPEGImagesなど）ごと丸ごとコピー
        shutil.copytree(src_voc2007, dst_voc2007)
        print("✅ 自動コピーが正常に完了しました！")
    except Exception as e:
        print(f"❌ コピー中にエラーが発生しました。パスを確認してください:\n{e}")
        sys.exit(1) # エラーが起きたらここで処理をストップ
else:
    print("ℹ️ Cドライブ側に既に VOC2007 フォルダが存在するため、コピーをスキップします。")
print("=" * 60)


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