import os
import json
import base64
import io
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """XMLをきれいなインデントに整形する"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def convert_and_arrange(json_dir_path, output_dir_path):
    # pathlib.Pathオブジェクトに変換
    json_dir = Path(json_dir_path)
    output_dir = Path(output_dir_path)
    
    # 事前チェック: 入力フォルダが存在するか
    if not json_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {json_dir}")
        print("パスが正しく設定されているか確認してください。")
        return

    # VOC形式のフォルダ構造を定義
    voc_base = output_dir / "VOCdevkit" / "VOC2007"
    annotations_dir = voc_base / "Annotations"
    jpeg_images_dir = voc_base / "JPEGImages"
    image_sets_dir = voc_base / "ImageSets" / "Main"
    
    # フォルダの自動作成
    annotations_dir.mkdir(parents=True, exist_ok=True)
    jpeg_images_dir.mkdir(parents=True, exist_ok=True)
    image_sets_dir.mkdir(parents=True, exist_ok=True)
    
    # JSONファイルを走査
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ 指定されたディレクトリにJSONファイルが見つかりません: {json_dir}")
        return

    image_ids = []
    print(f"📦 {len(json_files)} 個のJSONファイルを処理中...")

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ JSONの読み込み失敗 (スキップします): {json_path.name} -> {e}")
            continue
            
        base_name = json_path.stem
        xml_name = f"{base_name}.xml"
        jpg_name = f"{base_name}.jpg"
        
        # --- 1. 画像データの復元と保存 ---
        img = None
        if data.get('imageData'):
            try:
                img_data = base64.b64decode(data['imageData'])
                img = Image.open(io.BytesIO(img_data))
                img.save(jpeg_images_dir / jpg_name)
            except Exception as e:
                print(f"⚠️ 画像復元エラー (スキップします): {json_path.name} -> {e}")
                continue
        else:
            # imageDataがない場合は相対パスから取得を試みる
            img_path = json_dir / data.get('imagePath', '')
            if img_path.exists():
                img = Image.open(img_path)
                img.save(jpeg_images_dir / jpg_name)
            else:
                print(f"⚠️ 画像が見つかりません (スキップします): {json_path.name}")
                continue
                
        width, height = img.size
        depth = 3 if img.mode == 'RGB' else 1
        
        # --- 2. Pascal VOC形式のXML生成 ---
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = 'VOC2007'
        ET.SubElement(annotation, 'filename').text = jpg_name
        
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = str(depth)
        
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            if shape.get('shape_type', 'rectangle') == 'rectangle' and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                xmin = max(1, min(int(round(min(x1, x2))), width))
                ymin = max(1, min(int(round(min(y1, y2))), height))
                xmax = max(1, min(int(round(max(x1, x2))), width))
                ymax = max(1, min(int(round(max(y1, y2))), height))
                
                obj = ET.SubElement(annotation, 'object')
                ET.SubElement(obj, 'name').text = label
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(xmin)
                ET.SubElement(bndbox, 'ymin').text = str(ymin)
                ET.SubElement(bndbox, 'xmax').text = str(xmax)
                ET.SubElement(bndbox, 'ymax').text = str(ymax)
                
        xml_string = prettify(annotation)
        with open(annotations_dir / xml_name, 'w', encoding='utf-8') as xml_f:
            xml_f.write(xml_string)
            
        image_ids.append(base_name)

    # --- 3. YOLOX用管理テキスト（train.txt / val.txt）の自動生成 ---
    if image_ids:
        with open(image_sets_dir / "train.txt", "w", encoding='utf-8') as f_train:
            f_train.write("\n".join(sorted(image_ids)) + "\n")
        with open(image_sets_dir / "val.txt", "w", encoding='utf-8') as f_val:
            f_val.write("\n".join(sorted(image_ids)) + "\n")
        
    print(f"\n✨ 処理が正常に完了しました！")
    print(f"・正常に変換されたペア: {len(image_ids)} 組")
    print(f"・データセット保存先: {voc_base.resolve()}")

if __name__ == '__main__':
    # 💡 修正ポイント: 文字列の前に `r` を付与し、生文字列（Raw String）として扱います
    # これにより \t がタブ文字に変換されるのを防ぎます。
    
    JSON_INPUT_DIR = r"C:\Python_Venvs\train_images"
    VOC_OUTPUT_DIR = r"C:\Python_Venvs\env_YOLOX\YOLOX\datasets"
    
    convert_and_arrange(JSON_INPUT_DIR, VOC_OUTPUT_DIR)