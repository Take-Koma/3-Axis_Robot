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
# ==============================================================================

# 1つ上の階層に上がり、そこから「YOLOX」フォルダの絶対パスを作ります
yolox_dir = os.path.abspath(os.path.join(current_dir, "..", "YOLOX"))

# 作業ディレクトリを「YOLOX」フォルダに変更
os.chdir(yolox_dir)

# 実行したいコマンドをリスト形式で定義（各要素の末尾にコメントを追加）
cmd = [
    sys.executable,                         # 1. 実行中のPython（VSで選んだ仮想環境のpython.exe）
    "tools/train.py",                       # 2. YOLOXのメイン学習スクリプト
    "-f", "exps/default/yolox_nano_work.py",# 3. [-f] 使用するネットワーク構成ファイル（Expファイル）
    "-d", "1",                              # 4. [-d] 使用するGPU（デバイス）の数（1枚のGPUで学習）
    "-b", "4",                              # 5. [-b] トータルのバッチサイズ（1回にAIに学習させる画像枚数）
    "--fp16",                               # 6. 半精度（16bit浮動小数点）で計算して、メモリ節約＆高速化
    "-c", "yolox_nano.pth",                 # 7. [-c] 学習開始時にベースとして読み込む事前の重みファイル
    "-o"                                    # 8. [-o] GPUメモリを事前にガッツリ確保（占有）して効率化するフラグ
]

print("=" * 60)
print(f"🚀 YOLOX Training Started via YOLOX_CALL")
print(f"Working Dir: {yolox_dir}")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

# コマンドを実行（確認だけにしたいときは、頭に # をつけてコメントアウトしてください）
subprocess.run(cmd)


# -b, "4"（バッチサイズ）
# もしグラフィックボードのパワー（VRAM容量）に余裕があって、もっと学習をスピードアップさせたい場合は、ここを "8" や "16" に増やす。逆に、メモリ不足エラー（Out of Memory）が出た場合は数値を下げて調整する。

# -d, "1"（GPU数）
# 個人開発や一般的な開発PCであれば、グラフィックボードは1枚なので "1" で固定でOK

# --fp16（半精度演算）
# これが入っているおかげで、通常の32bit計算よりも圧倒的に省メモリかつ高速に学習が回る。YOLOX-Nanoのような軽量モデルをサクッと回すには必須の神オプション。