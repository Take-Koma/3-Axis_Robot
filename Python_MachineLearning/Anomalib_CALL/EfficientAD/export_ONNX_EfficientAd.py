import os
from pathlib import Path
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.deploy import ExportType

def main():
    # 1. 学習で生成された .ckpt ファイルのパスを指定します
    ckpt_file_path = Path(".../results/EfficientAd/Test_Product/latest/weights/lightning/model.ckpt").resolve() 

    if not ckpt_file_path.exists():
        print(f"エラー: {ckpt_file_path} が見つかりません。パスを確認してください。")
        return

    # 2. ONNXを出力する保存先フォルダを指定
    export_dir = Path("./exported_onnx")
    export_dir.mkdir(exist_ok=True)

    # 3. モデルの「空の枠組み」を用意（学習時と同じEfficientADを指定）
    model = EfficientAd()

    # 4. 実行エンジンの準備
    engine = Engine()

    # 5. .ckpt を読み込みながら、一気にONNXへエクスポート！
    print("=== ONNXへの変換を開始します ===")
    try:
        exported_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_file_path,  # ← ここで学習済み重みを注入します
            input_size=(256, 256)   # ここで入力サイズを指定して固定化します（TwinCATは可変サイズを許さないため）   (EfficientADは通常 256x256 で処理されます)
        )
        print("=== エクスポート大成功！ ===")
        print(f"出力先: {exported_path}")
    except Exception as e:
        print(f"エクスポート中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()



 
# 仮想環境(env_Anomalib)でコマンドプロンプトにて実行の事

# 1. TwinCATが読めるようにグラフをシンプルにする（事前に pip install onnx-simplifier を実行）
# python -m onnxsim exported_onnx/model_fixed.onnx exported_onnx/model_for_twincat.onnx