import sys
import os
import argparse
import torch
import cv2
import numpy as np

# 将当前路径添加到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从绝对路径导入 DetectMultiBackend
from models.common import DetectMultiBackend  # 确保该路径指向正确的模块
from models.export import export_formats

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """Resize image to a rectangular shape while maintaining aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # calculate the ratio to resize
    if not scaleup:  # if scale up is False, prevent enlarging
        ratio = min(ratio, 1.0)
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))  # new size without padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # calculate padding
    if auto:  # if auto padding is enabled
        dw, dh = dw / 2, dh / 2  # divide padding by 2
    elif scaleFill:  # if scale fill is enabled
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
    else:  # if no auto or scale fill
        dw, dh = np.floor(dw), np.floor(dh)  # apply floor to padding
    # resize and pad image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # calculate top and bottom padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # calculate left and right padding
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add padding
    return img

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Apply NMS to prediction results."""
    # Your NMS implementation here
    # This is a simplified version for illustration
    output = []
    for pred in prediction:
        if pred is not None and len(pred):
            # Process prediction here...
            output.append(pred)  # Placeholder
    return output

import cv2
import numpy as np

def resize_image(image, target_size=(640, 640)):
    """Resize image to the target size and return the resized image and original dimensions."""
    original_shape = image.shape[:2]  # 记录原始图像的形状
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)  # 调整为640x640
    return resized_image, original_shape

# 加载模型并进行推理
def detect(image_path, model_path='fracture_detection/yolov5m_fracture.pt', conf_threshold=0.5, input_size=640):
    model = DetectMultiBackend(model_path)
    
    # 预处理图片
    img, original_shape = preprocess_image(image_path, input_size)

    # 模型推理
    pred = model(img)

    # 非极大值抑制（NMS）去除多余的检测框
    pred = non_max_suppression(pred, conf_threshold)

    # 读取原图大小用于坐标转换
    original_img = Image.open(image_path)
    original_size = original_img.size  # 原图的宽高
    original_width, original_height = original_size  # 解包宽高

    # 坐标转换
    boxes = []
    if pred[0] is not None and len(pred[0]):  # 如果有检测结果
        # 获取输入图像的宽高
        input_height, input_width = img.shape[2], img.shape[3]

        # 计算宽高比例
        h_ratio = original_height / input_height
        w_ratio = original_width / input_width

        # 计算左边距
        pad_x = (input_width - original_width * min(input_width / original_width, input_height / original_height)) / 2
        pad_y = (input_height - original_height * min(input_width / original_width, input_height / original_height)) / 2

        # 右移的比例，设置为图像宽度的 33%
        shift_x = int(original_width * 0)  # 将检测框右移原图宽度的 33%

        for detection in pred[0]:
            x1, y1, x2, y2 = detection[:4].tolist()

            # 进行坐标转换
            x1 = int((x1 - pad_x) * w_ratio) + shift_x  # 右移
            y1 = int(y1 * h_ratio)
            x2 = int((x2 - pad_x) * w_ratio) + shift_x  # 右移
            y2 = int(y2 * h_ratio)

            # 限制坐标在图像边界内
            x1 = min(max(x1, 0), original_width - 1)
            x2 = min(max(x2, 0), original_width - 1)

            boxes.append([x1, y1, x2, y2])  # 存储为整数类型的坐标

    return boxes, original_img




def main():
    parser = argparse.ArgumentParser(description='Fracture Detection with YOLOv5')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    boxes = detect(args.image_path)
    for det in boxes:
        if det is not None:
            for *xyxy, conf, cls in det:
                print(f'Detected: {cls} with confidence {conf} at {xyxy}')

if __name__ == "__main__":
    main()
