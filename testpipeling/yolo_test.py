# detection.py
from ultralytics import YOLOv10
import cv2
import numpy as np
from sklearn.cluster import KMeans

class DetectionProcessor:
    def __init__(self, model_path):
        """初始化YOLOv10模型"""
        self.model = YOLOv10(model_path)
        self.model_names = self.model.names

    def _detect_center(self, cropped_img):
        """内部使用的中心点检测方法"""
        params = {
            'color_space': 'LAB',
            'channel_index': 0,
            'clahe_clip': 2.0,
            'threshold_type': 'adaptive',
            'morph_kernel': (3,3),
            'cluster_num': 1
        }

        try:
            # 颜色空间转换
            if params['color_space'] == 'HSV':
                cvt_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            elif params['color_space'] == 'LAB':
                cvt_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
            else:
                cvt_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2YCrCb)

            # 通道处理和增强
            channel = cvt_img[:, :, params['channel_index']]
            clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8,8))
            enhanced = clahe.apply(channel)

            # 阈值处理
            if params['threshold_type'] == 'adaptive':
                binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 21, 5)
            else:
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params['morph_kernel'])
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 聚类分析
            y_coords, x_coords = np.where(cleaned == 0)
            if len(x_coords) < 10:
                return (-1, -1)

            if len(x_coords) > 5000:
                indices = np.random.choice(len(x_coords), 5000, replace=False)
                x_coords = x_coords[indices]
                y_coords = y_coords[indices]

            coordinates = np.column_stack((x_coords, y_coords))
            kmeans = KMeans(n_clusters=params['cluster_num'], n_init=10).fit(coordinates)
            return tuple(map(int, kmeans.cluster_centers_[0]))

        except Exception as e:
            print(f"Center detection error: {str(e)}")
            return (-1, -1)

    def process(self, img_path):
        """主处理方法"""
        result = {
            'log_info': [],
            'detections': []
        }

        original_img = cv2.imread(img_path)
        if original_img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

        # YOLO检测
        yolo_results = self.model.predict(original_img, imgsz=640, conf=0.15, iou=0.5)

        for box_idx, box in enumerate(yolo_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0].item())
            class_name = self.model_names[class_id]

            # 裁剪检测区域
            cropped = original_img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            # 中心点检测
            center = self._detect_center(cropped)
            if center == (-1, -1):
                continue

            # 坐标转换
            global_center = (x1 + center[0], y1 + center[1])
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # 构建结果
            detection_data = {
                'label': class_name,
                'bbox': (x1, y1, x2, y2),
                'box_center': box_center,
                'obj_center': global_center
            }

            # 生成日志信息
            log_entries = [
                f"检测框 {box_idx} ({class_name})",
                f"检测框坐标: ({x1}, {y1}, {x2}, {y2})",
                f"几何中心: {box_center}",
                f"实际中心: {global_center}"
            ]

            result['log_info'].extend(log_entries)
            result['detections'].append(detection_data)

        return result
