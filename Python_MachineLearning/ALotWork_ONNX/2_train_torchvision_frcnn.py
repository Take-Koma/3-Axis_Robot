import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
import torchvision
from torchvision.transforms import functional as F

# ==========================================================================================
# 固定パラメータ定義（PLC側の前処理・推論ロジックと1ミリのズレもなく完全一致させる）
# ==========================================================================================
INPUT_WIDTH = 640                  # TwinCAT側でリサイズする幅と完全一致
INPUT_HEIGHT = 640                 # TwinCAT側でリサイズする高さと完全一致
MAX_DETECTIONS = 50                # 1画面あたりの最大検出数（TwinCAT側の配列要素数50と一致）


class CocoBBoxDataset(Dataset):
    """
    【クラス説明】
    coco_bbox.jsonを読み込み、画像を【640x640】に強制統一しながら、
    アノテーションされた正解の四角（BBox）の座標もその比率に合わせて自動換算する
    「TwinCAT Vision専用」のデータ配給クラス。
    """
    def __init__(self, images_dir, coco_json_path):
        self.images_dir = Path(images_dir)
        self.coco = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))

        # 画像情報とアノテーション（四角形）情報の整理
        self.imgs = list(self.coco["images"])
        self.imgid_to_anns = {}
        for ann in self.coco["annotations"]:
            self.imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_path = self.images_dir / img_info["file_name"]
        
        # 画像をRGBカラー形式で展開
        image = Image.open(img_path).convert("RGB")
        
        # ★【超重要】元の画像サイズ（例: 816x624など）を記憶しておく
        orig_w, orig_h = image.size

        anns = self.imgid_to_anns.get(img_info["id"], [])
        boxes = []
        labels = []

        for a in anns:
            x, y, w, h = a["bbox"]
            
            # ★【超重要】画像が640x640に潰されるので、BBoxの座標[x1, y1, x2, y2]も
            # 元の解像度から640基準の比率へと事前に換算して、学習のズレを無くす！
            x1 = (x / orig_w) * float(INPUT_WIDTH)
            y1 = (y / orig_h) * float(INPUT_HEIGHT)
            x2 = ((x + w) / orig_w) * float(INPUT_WIDTH)
            y2 = ((y + h) / orig_h) * float(INPUT_HEIGHT)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(a["category_id"])  # 1番（work）

        # PyTorch計算用のテンソル型（Float32 / Int64）へ変換
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # AIに渡すターゲット（正解データ）の辞書を作成
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]]),
            "area": torch.tensor([a["area"] for a in anns], dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64), # 0固定
        }

        # ★【超重要】すべての画像をあらかじめ640x640へ強制リサイズ！
        # これによりONNX書き出し時の入力レイヤーが「可変長」になるのを根本から防ぎます
        image = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
        
        # 画像のピクセル（0〜255）を、AIが処理しやすい（0.0〜1.0）へスケール変換
        image = F.to_tensor(image)

        return image, target


def collate_fn(batch):
    """
    【関数説明】
    画像と正解データのサイズを綺麗に束ねてDataLoaderに引き渡す補助関数。
    """
    return tuple(zip(*batch))


def main():
    # --------------------------------------------------------------------------------------
    # 1. 演算デバイスの設定（T600 GPUが使えればCuda、なければCPU）
    # --------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"【使用するデバイス】: {device}")
    if device.type != "cuda":
        print("【警告】GPU（CUDA）が認識されていません。CPUだと学習にかなり時間がかかります！")

    # --------------------------------------------------------------------------------------
    # 2. データの準備（リサイズ機能付き自作Datasetの適用）
    # --------------------------------------------------------------------------------------
    dataset = CocoBBoxDataset("dataset_raw/images", "coco_bbox.json")
    
    # batch_size=2 で2枚ずつ小分けにしてAIに学習させる
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # --------------------------------------------------------------------------------------
    # 3. AIモデルの組み立て（Faster R-CNN + MobileNetV3）※超重要リフォーム
    # --------------------------------------------------------------------------------------
    print("【モデル準備中】ベースとなるモデルをロード＆TwinCAT最適化パラメータへ書き換えています...")
    
    # ❌ デフォルトのままだと、モデルの内部で勝手に長辺1333/短辺800ピクセルに拡大するお節介機能（自動リサイズ）
    # が働き、これがTwinCAT Vision Serviceをパニックに陥らせて3.5GBのメモリ爆発を引き起こします。
    # ⭕ 引数で min_size/max_size を一律640に固定し、自動リサイズを完全に封印（無効化）します！
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT",
        image_mean=[0.0, 0.0, 0.0],       # モデル内部での平均値の引き算をパス（PLC側で一括管理）
        image_std=[1.0, 1.0, 1.0],        # モデル内部での標準偏差の割り算をパス
        min_size=INPUT_WIDTH,             # 内部での自動拡大リサイズの下限を640に制限
        max_size=INPUT_HEIGHT,            # 内部での自動拡大リサイズの上限を640に制限
        box_detections_per_img=MAX_DETECTIONS  # 1画面あたりの最大検出数を「50個」に厳格に固定！
    )
    
    # 出力ヘッドを「背景(0)」と「work(1)」の合計2クラス用にリフォーム
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, 
        num_classes=2
    )

    # カスタマイズした軽量モデルをGPU（またはCPU）メモリに展開
    model.to(device)
    # モデルを「学習モード（ロス計算有効化）」に切り替え
    model.train()

    # --------------------------------------------------------------------------------------
    # 4. 最適化アルゴリズム（AIのパラメータ修正係）の設定
    # --------------------------------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # --------------------------------------------------------------------------------------
    # 5. 学習ループスタート
    # --------------------------------------------------------------------------------------
    num_epochs = 30  # 全18枚の画像を30周ループさせて覚え込ませる
    print("【学習開始】AIのトレーニングをスタートします。")
    print("--------------------------------------------------")

    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for images, targets in data_loader:
            # 画像と正解データを指定デバイス（GPU/CPU）に転送
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # AIに予測を実行させ、ズレ（Loss：損失）の合計を計算
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())

            # 誤差逆伝播（バックプロパゲーション）で脳みその重みを賢い方向へ微調整
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        # 1周（1エポック）ごとに平均のエラー率（Loss）を表示（これが下がっていけば大成功）
        print(f"周回 (Epoch) [{epoch+1}/{num_epochs}] - エラー率 (Loss): {total_loss/len(data_loader):.4f}")

    # --------------------------------------------------------------------------------------
    # 6. 賢くなったモデル重みファイルの保存
    # --------------------------------------------------------------------------------------
    print("--------------------------------------------------")
    save_path = "work_frcnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"【大成功】固定サイズ学習が完了しました！ベース重みを保存しました: {save_path}")
    print("※ 次は、このpthファイルとTopKWrapperを合体させて、dynamic_axes無しの完全固定サイズONNXへエクスポートしてください！")


if __name__ == "__main__":
    main()