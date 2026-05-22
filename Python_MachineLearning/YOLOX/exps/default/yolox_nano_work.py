import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # クラス数: 今回は1クラス（変更があれば適宜修正してください）
        self.num_classes = 1
        # 💡【重要】推論時に使用するクラス名を明示的に登録する
        self.class_names = ("work",)

        # 💡【これを追加】学習と評価の画像サイズを640x640に固定する
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # モデルの規模: Nanoの設定値
        self.depth = 0.33
        self.width = 0.25
        # 出力ディレクトリ名
        self.exp_name = "yolox_nano_work"
        # 学習エポック数
        self.max_epoch = 300
        # データロード用のプロセス数（PCスペックに合わせて調整）
        self.data_num_workers = 2
        # ログ出力間隔
        self.print_interval = 10
        # 評価（テスト）を行うエポック間隔
        self.eval_interval = 10
        
        # 💡 【重要】データセットのルートパスを絶対パスで指定
        # YOLOXが探すのは「VOCdevkit」の「親フォルダ」です
        self.data_dir = r"C:\Python_Venvs\env_YOLOX\YOLOX\datasets\VOCdevkit"

    def get_dataset(self, cache=False, cache_type="ram"):
        """学習用データセットを準備するメソッド"""
        from yolox.data import VOCDetection, TrainTransform
        
        # VOCDetectionはフォルダ構造を厳密にチェックします
        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=[('2007', 'train')],  # 2007年データのtrainセットを使用
            dataset_name="work",
            img_size=self.input_size,
            # 学習時に画像を加工（Mosaic拡張など）するための変換設定
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """評価（検証）用データセットのローダーを準備するメソッド"""
        import torch
        from yolox.data import VOCDetection, ValTransform

        # 評価用のデータセット定義
        valdataset = VOCDetection(
            data_dir=self.data_dir,
            image_sets=[('2007', 'train')], # 評価にもtrainデータを使用（必要に応じて'val'に変更）
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        # 分散学習かどうかでサンプラーを切り替え
        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        
        # ローダーを作成して返却
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """評価実行用（VOCEvaluator）を作成するメソッド"""
        from yolox.evaluators import VOCEvaluator

        # 上記で作ったデータローダーを取得
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        
        # mAP計算などを行う評価クラスの初期化
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf, # 予測の信頼度閾値
            nmsthre=self.nmsthre,   # NMSの閾値
            num_classes=self.num_classes,
        )
        return evaluator