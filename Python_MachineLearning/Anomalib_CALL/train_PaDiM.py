import os
from pathlib import Path
from anomalib.data import Folder  # 自前画像を使うためのモジュール
from anomalib.models import Padim
from anomalib.engine import Engine

def main():
    # 1. データの読み込み設定
    dataset_root = Path("../Python_MachineLearning/my_dataset").resolve()
    
    datamodule = Folder(
        name="my_product",
        root=dataset_root,
        normal_dir="train/good",
        abnormal_dir="train/good",
        normal_test_dir="train/good",
        # 修正①: "none"ではなく、テスト用と同じデータを検証用にも使い回す（ゼロ除算も回避できます）
        val_split_mode="same_as_test", 
    )

    print(f"フォルダの存在確認: {dataset_root.exists()}")
    import glob
    files = list(dataset_root.glob("train/good/*"))
    print(f"画像枚数: {len(files)}")
    if len(files) == 0:
        print("警告: 画像が見つかりません！パスを見直してください。")

    datamodule.setup()

    # 2. AIモデル（Padim）と実行エンジンの準備
    model = Padim()
    
    # 修正②: Engineに「検証（val）は一切やらなくていい」と強制的に指示し、タスクを明示する
    engine = Engine(
        limit_val_batches=0,   # バリデーションを0回にする
        num_sanity_val_steps=0 # 学習前のチェックもスキップ
    )

    # 3. 学習（正常な状態を覚え込ませる）
    print("=== 学習を開始します ===")
    engine.fit(model=model, datamodule=datamodule)
    print("=== 学習が完了しました ===")

    # 4. 判定テストを実行して結果を出力する
    print("=== 推論テストと結果保存を実行します ===")
    engine.test(model=model, datamodule=datamodule)
    print("=== すべての処理が完了しました！ ===")
    print("Anomalib_CALL フォルダ内に自動生成された 『results』 フォルダを確認してください。")

if __name__ == "__main__":
    main()