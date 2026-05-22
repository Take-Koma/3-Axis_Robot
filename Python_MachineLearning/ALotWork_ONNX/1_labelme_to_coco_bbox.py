
import json
import os
from pathlib import Path
from PIL import Image

def rect_to_coco_bbox(points):
    """
    【関数説明】
    Labelmeの四角形データ（左上と右下の2点座標）を、
    AIが読み込める「左上のX, 左上のY, 横幅, 縦幅」の形に計算し直す関数。
    """
    (x1, y1), (x2, y2) = points
    
    # マウスを逆方向にドラッグして囲んでもバグらないように、小さい値を「左上(min)」とする
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    
    # 四角形の「横幅」と「縦幅」を、絶対値（マイナスにしない）で計算
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    return [x_min, y_min, w, h]

def convert(images_dir, labels_dir, out_json, category_name="work"):
    """
    【関数説明】
    バラバラのLabelmeのJSONファイルを読み込んで、
    1つの大きな「COCO形式」のJSONファイルに合体させるメイン処理。
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # COCO形式の空っぽの「台帳」を用意する
    coco = {
        "images": [],       # 画像のファイル名や縦横サイズを記録するリスト
        "annotations": [],  # 囲った四角形の座標を記録するリスト
        "categories": [{"id": 1, "name": category_name}] # 見つけたい物の名前（1番: work）
    }

    ann_id = 1  # 四角形1個ずつに割り振る通し番号（ID）
    img_id = 1  # 画像1枚ずつに割り振る通し番号（ID）

    # imagesフォルダの中にあるファイルを1つずつループで処理
    for img_path in sorted(images_dir.glob("*")):
        # 画像ファイル以外（余計なシステムファイルなど）は無視してスキップ
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        # 画像に対応するラベル（JSON）のパスを計算（例: 0001.jpg -> 0001.json）
        label_path = labels_dir / (img_path.stem + ".json")
        
        # もしラベルファイルが存在しなかったら、その画像はスキップ
        if not label_path.exists():
            print(f"【確認】ラベルが見つからないためスキップします: {img_path.name}")
            continue

        # 画像を実際に開いて、縦と横のピクセル解像度を取得する
        with Image.open(img_path) as im:
            width, height = im.size

        # 台帳の「images」の欄に、画像情報を書き込む
        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        # Labelmeが作ったJSONファイルをテキストとして読み込む
        data = json.loads(label_path.read_text(encoding="utf-8"))
        
        # JSONの中にある「shapes（マウスで描いた図形たち）」を1個ずつループで処理
        for shp in data.get("shapes", []):
            # もしラベル名が「work」じゃなかったら無視する（打ち間違い対策）
            if shp.get("label") != category_name:
                continue
            # もし図形が「rectangle（長方形）」じゃなかったら無視する
            if shp.get("shape_type") != "rectangle":
                continue

            # 図形の座標データを、[X, Y, 幅, 高さ] の形に変換
            bbox = rect_to_coco_bbox(shp["points"])
            
            # 四角形の面積を計算（AIの評価用）
            area = bbox[2] * bbox[3]

            # 台帳の「annotations」の欄に、四角形の座標と「どの画像のやつか（img_id）」を書き込む
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1, # 1番（work）
                "bbox": bbox,     # 計算した [X, Y, 幅, 高さ]
                "area": area,     # 面積
                "iscrowd": 0      # 1つの塊として扱わないための設定（0固定）
            })
            ann_id += 1 # 四角形のIDを1進める

        img_id += 1 # 画像のIDを1進める

    # 完成した台帳を、指定されたファイル名（coco_bbox.json）で書き出す
    Path(out_json).write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"==================================================")
    print(f"【大成功】合体ファイルを作成しました: {out_json}")
    print(f"==================================================")

if __name__ == "__main__":
    # ここでフォルダの場所と、出力するファイル名を指定しています
    convert(
        images_dir="dataset_raw/images",
        labels_dir="dataset_raw/labels",
        out_json="coco_bbox.json",
        category_name="work"
    )