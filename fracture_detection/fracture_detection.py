import torch
from PIL import Image
import cv2
import numpy as np
import sys
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
import streamlit as st

# 设置 YOLOv5 的本地路径
yolov5_path = 'fracture_detection/yolov5-master'  # 替换为你克隆的 yolov5 仓库的路径
sys.path.append(yolov5_path)

# 预处理图片
def preprocess_image(image_path, input_size=640):
    img = Image.open(image_path).convert('RGB')
    original_shape = img.size  # 记录原始图像的形状
    img = np.array(img)  # 转换为 numpy 数组
    img = letterbox(img, new_shape=(input_size, input_size))[0]  # 调整大小并保持比例，获取调整后的图像
    img = img.transpose((2, 0, 1))  # 转换为模型需要的 [C, H, W] 格式
    img = np.ascontiguousarray(img)  # 确保内存连续
    img = torch.from_numpy(img).float() / 255.0  # 归一化处理并转换为张量
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 添加批次维度
    return img, original_shape  # 返回处理后的图像和原始形状


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



# 结果可视化
def visualize_detection(original_img, boxes):
    img = np.array(original_img)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 画出检测框
    return img

# Streamlit 界面
def main():
    st.title("骨折检测")
    
    # 上传图片
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 将上传的图片转换为 PIL 格式
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 检测骨折
        boxes, original_shape = detect(image_path)  # 修改为调用新的 detect 函数
        
        # 可视化检测结果
        original_img = Image.open(image_path)  # 读取原始图片
        result_img = visualize_detection(original_img, boxes)

        # 显示检测结果
        st.image(result_img, caption="检测结果", use_column_width=True)

if __name__ == "__main__":
    main()
