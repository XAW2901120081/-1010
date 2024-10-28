import torch
from PIL import Image
import cv2
import numpy as np
import sys
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
import streamlit as st
import os
import zipfile
from io import BytesIO
import shutil

# 设置 YOLOv5 的本地路径
yolov5_path = 'fracture_detection/yolov5-master'  # 替换为你克隆的 yolov5 仓库的路径
sys.path.append(yolov5_path)

# 定义上传路径和输出路径
UPLOAD_FOLDER = "uploads/"
OUTPUT_FOLDER = "output/"

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 预处理图片
def preprocess_image(image_path, input_size=640):
    img = Image.open(image_path).convert('RGB')
    original_shape = img.size  # 记录原始图像的形状
    img = np.array(img)  # 转换为 numpy 数组
    img = letterbox(img, new_shape=(input_size, input_size))[0]  # 调整大小并保持比例
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

        # 右移的比例，设置为图像宽度的 33%
        shift_x = int(original_width * 0)  # 如果需要右移框位置可以调整这个值

        for detection in pred[0]:
            x1, y1, x2, y2 = detection[:4].tolist()

            # 进行坐标转换
            x1 = int(x1 * w_ratio) + shift_x  # 右移
            y1 = int(y1 * h_ratio)
            x2 = int(x2 * w_ratio) + shift_x  # 右移
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

# 保存上传的图片
def save_uploadedfile(uploadedfile):
    with open(os.path.join(UPLOAD_FOLDER, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join(UPLOAD_FOLDER, uploadedfile.name)

# 压缩检测结果为 zip 文件
def zip_output(output_folder):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for foldername, subfolders, filenames in os.walk(output_folder):
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                zf.write(filepath, arcname=os.path.relpath(filepath, output_folder))
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit 界面
def main():
    st.title("批量骨折检测")

    # 上传多个图片
    uploaded_files = st.file_uploader("选择多张图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"已上传 {len(uploaded_files)} 张图片")

        # 批量检测
        if st.button("开始批量检测"):
            st.write("正在进行骨折检测...")

            for uploaded_file in uploaded_files:
                image_path = save_uploadedfile(uploaded_file)  # 保存每张上传的图片

                # 检测骨折
                boxes, original_img = detect(image_path)
                
                # 可视化检测结果并保存
                result_img = visualize_detection(original_img, boxes)
                result_img_path = os.path.join(OUTPUT_FOLDER, uploaded_file.name)
                Image.fromarray(result_img).save(result_img_path)

                st.image(result_img, caption=f"{uploaded_file.name} 检测结果", use_column_width=True)
            
            st.success("批量检测完成！")

            # 提供下载压缩包
            zip_file = zip_output(OUTPUT_FOLDER)
            st.download_button(label="下载所有检测结果", data=zip_file, file_name="detection_results.zip", mime="application/zip")

            # 清理上传和输出文件夹（可选）
            if st.button("清理文件"):
                shutil.rmtree(UPLOAD_FOLDER)
                shutil.rmtree(OUTPUT_FOLDER)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                st.success("文件已清理")

if __name__ == "__main__":
    main()
