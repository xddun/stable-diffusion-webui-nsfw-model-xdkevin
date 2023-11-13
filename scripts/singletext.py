import os
import cv2
import numpy as np
import onnxruntime

# Load ONNX model
onnx_model_path = './model_fp16.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Example usage:
image_path = r"C:\Users\xjsd\Pictures\fangzhu\1.jpg"
# Load image using OpenCV
image = cv2.imread(image_path)
if image is None:
    print("Image Load Failure: ", image_path)
    exit(-1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image = cv2.resize(image, (299, 299))
image = image.astype(np.float16) / 255.0
image = np.expand_dims(image, axis=0)

# Perform inference
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
# Run the ONNX model
res_list = ort_session.run([output_name], {input_name: image})[0].tolist()[0]
print(res_list)
# 打印最大值的下标和名称
cls_name = ["drawings", "hentai", "neutral", "porn", "sexy"]
# 求最大值的下标
index = np.argmax(res_list)
# 带上类别名字
print(cls_name[index],res_list[index])


