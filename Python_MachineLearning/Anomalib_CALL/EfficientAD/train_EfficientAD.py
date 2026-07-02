import os
from pathlib import Path

from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models import EfficientAd
 
IMAGE_SIZE = (256, 256)

# 1. データの読み込み設定
dataset_root = Path(".../my_dataset").resolve()
 
datamodule = Folder(
    name="Test_Product",
    root=dataset_root,
    normal_dir="train/good",
    normal_test_dir="test/good",
    abnormal_dir="test/bad",
    train_batch_size=1, # EfficientADはtrain_batch_size=1が必須、1以外にするとEngine.fit()の途中でエラーになります
    eval_batch_size=8,
    num_workers=0, # データローディングの並列数（環境に合わせて調整してください）4だと「DataLoader worker (pid(s) 22888, 33348, 22824, 18052) exited unexpectedly」と出てエラーになるので、0にしてシングルスレッドで動かしてます。
)
 
model = EfficientAd(
    imagenet_dir=r"C:/Python_Venvs/imagenette", # EfficientADは学習前にImageNetの特徴をダウンロードしてくる必要があるため、その保存先を指定します。ローカルにダウンロード済みならそのパスを、まだなら適当な空フォルダを指定してください（初回実行時に自動でダウンロードされます）。
) 

engine = Engine(
    accelerator="gpu",   # GPUなければ "cpu"
    devices=1,
    max_epochs=20,       # まずは50程度（PoCなら20でも可）
)
 
engine.fit(model=model, datamodule=datamodule)
 
# 推論（まずPythonで動作確認）
preds = engine.predict(model=model, datamodule=datamodule)
print(type(preds), len(preds))




# 後々画像サイズを変更したい際に追加項目ここから

# 256x256の前処理オブジェクトを生成
# my_pre_processor = EfficientAd.configure_pre_processor(image_size=(256, 256))

# モデルにそれを直に放り込む
#model = EfficientAd(
#    imagenet_dir=r"C:/Python_Venvs/imagenette",
#    pre_processor=my_pre_processor  # ← ここに渡す
#)

# 後々画像サイズを変更したい際に追加項目ここまで





# 1) 重要：anomalibのモデルは「全部がONNX向き」ではありません
# anomalibには大きく2系統あります。
# ONNXにしやすい（=ネットワーク推論で完結しやすい）モデル
# FastFlow（Flow系）
# STFPM（教師ありではなく特徴再構成系）
# DRAEM（再構成＋差分系）
# EfficientAD（軽量で運用向きのことが多い）
# → これらは「forwardがTensorを返す」形に整えやすく、固定入力/固定出力でONNX化しやすいです。
# ONNXにしにくいモデル（注意）
# PatchCore / PaDiM など、推論時に メモリバンク/最近傍探索/統計推定を強く使うもの
# → “学習後の追加処理込み”で成立することが多く、ONNX単体に閉じにくいです（TwinCATで完結が難しくなりがち）。
# TwinCATに載せる前提なら、最初から FastFlow / EfficientAD あたりを推奨します。
 
# EfficientAD / FastFlow 
# この二つがTF3820でも使えそうな感じがするONNXを出力できる。

# EfficientADは軽量で運用向き、FastFlowは高精度で大規模なモデルが多い傾向があります。



