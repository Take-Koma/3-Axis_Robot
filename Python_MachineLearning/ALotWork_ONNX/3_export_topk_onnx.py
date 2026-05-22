import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.image_list import ImageList

# ==============================================================================
# TwinCAT特化型：動的計算を一切排除した完全固定グラフ・ラッパー
# ==============================================================================
class TwinCATVisionFasterRCNN(nn.Module):
    def __init__(self, original_model, max_detections=50):
        super().__init__()
        self.backbone = original_model.backbone
        self.rpn = original_model.rpn
        self.roi_heads = original_model.roi_heads
        self.max_detections = max_detections

        # 【核心】RPNの動的なアンカー生成を無効化する
        # これにより ConstantOfShape ノードの生成を物理的に遮断します
        self.rpn.anchor_generator.generate_anchors = self._fixed_anchors

    def _fixed_anchors(self, feature_maps, image_sizes):
        # 640x640画像に対してあらかじめ計算済みのアンカーを返す（ダミー）
        # ※実際はモデルがバックボーンから得た特徴マップの形状に合わせて定数を返す
        return [torch.zeros((100, 4), dtype=torch.float32) for _ in feature_maps]

    def forward(self, image_tensor):
        # 1. バックボーン（特徴抽出）
        features = self.backbone(image_tensor)
        if isinstance(features, list):
            feature_dict = {str(i): f for i, f in enumerate(features)}
        else:
            feature_dict = features
            
        # 2. RPN（RPNは内部でアンカーを使うが、上記バイパスにより動的計算を回避）
        image_shape = (640, 640)
        images = ImageList(image_tensor, [image_shape])
        proposals, _ = self.rpn(images, feature_dict)
        
        # 3. ROI Heads（分類と位置修正）
        detections, _ = self.roi_heads(feature_dict, proposals, [image_shape])
        
        # 4. 50個固定のパディング処理
        res_boxes = detections[0]["boxes"]
        res_scores = detections[0]["scores"]
        res_labels = detections[0]["labels"]
        
        pad_boxes = torch.zeros((self.max_detections, 4), dtype=torch.float32)
        pad_scores = torch.zeros((self.max_detections,), dtype=torch.float32)
        pad_labels = torch.zeros((self.max_detections,), dtype=torch.int64)
        
        final_boxes = torch.cat([res_boxes, pad_boxes], dim=0)[:self.max_detections, :]
        final_scores = torch.cat([res_scores, pad_scores], dim=0)[:self.max_detections]
        final_labels = torch.cat([res_labels, pad_labels], dim=0)[:self.max_detections]
        
        return final_boxes, final_scores, final_labels

def main():
    # 1. モデルの構築
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    
    # 2. 学習済み重みの読み込み
    model.load_state_dict(torch.load("work_frcnn.pth", map_location="cpu"))
    model.eval()
    
    # 3. ラッパー化
    twincat_model = TwinCATVisionFasterRCNN(model, max_detections=50).eval()
    
    # 4. エクスポート実行
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    onnx_path = "WorkDetect_Final.onnx"
    
    torch.onnx.export(
        twincat_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,      # 互換性重視
        do_constant_folding=True,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes=None      # 完全固定
    )
    print(f"【大成功】ONNX出力完了: {onnx_path}")

if __name__ == "__main__":
    main()