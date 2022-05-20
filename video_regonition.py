import cv2
import face_recognition
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
jnk_image = face_recognition.load_image_file("jnk.jpg")
jnk_face_encoding = face_recognition.face_encodings(jnk_image)[0]
obama_image = face_recognition.load_image_file("ob.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
known_face_encodings = [
    jnk_face_encoding,
    obama_face_encoding
]
known_face_names = [
    "jnk",
    "obm"
]
# 获取摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 打开摄像头
cap.open(0)

while cap.isOpened():
    # 获取画面
    flag, frame = cap.read()
    unknown_image = face_recognition.load_image_file(frame)
    face_locations = face_recognition.face_locations(unknown_image)
    # 人脸特征提取
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 判断和哪张人脸匹配
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        # 提取和未知人脸距离最小的已知人脸编号
        best_match_index = np.argmin(face_distances)
        # 提取匹配的已知人脸名
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # 为人脸画边界框
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # 在人脸边界框下方绘制该人脸所属人的名字
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        del draw
        cv2.imshow('mytest', pil_image)
            # 设置退出按钮
        key_pressed = cv2.waitKey(100)
        print('单机窗口，输入按键，电脑按键为', key_pressed, '按esc键结束')
    if key_pressed == 27:
        break
# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()