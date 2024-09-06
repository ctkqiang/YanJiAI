# -*- coding: UTF-8 -*-
try:
    import os
    import cv2
    import time
    import torch
    import numpy as np
    import torchvision.transforms as transforms
    from datetime import datetime
    from PIL import ImageFont, ImageDraw, Image
    from module.colors import *
    from module.trainer import *
    from deepface import DeepFace
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
except ImportError:
    raise ImportError("🥹无法安装配件")
finally:
    pass


class 眼迹AI:

    def __init__(self) -> None:
        """
        初始化眼迹AI类。

        Attributes:
            名称 (str): 眼迹AI的名称。
            粗细 (float): 画笔粗细。
            字体路径 (str): 字体文件路径。
            眼部分类器: OpenCV眼部分类器对象。
            眼睛分类器: OpenCV眼睛分类器对象。
            捕获: OpenCV视频捕获对象。
            眼睛位置 (dict): 指示眼睛位置的文本和颜色的字典。
            初始帧数 (float): 初始帧数值用于计算眨眼。
        """
        super(眼迹AI, self).__init__()

        self.名称: str = "眼迹AI"
        self.粗细: float = 0.5
        self.字体路径: str = "assets/YRDZST Medium.ttf"
        self.enforce_detection: bool = False

        # 加载预训练的眼部和眼睛分类器
        self.眼部分类器 = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.眼睛分类器 = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # 初始化视频捕获
        self.捕获 = cv2.VideoCapture(0)
        self.可用摄像头 = []
        self.索引 = 0

        # 定义眼睛位置对应的文本和矩形颜色的字典
        self.眼睛位置: dict = {
            "左": ("向左看", (255, 0, 0)),  # 红色
            "右": ("向右看", (255, 0, 0)),  # 蓝色
            "正": ("正视", (0, 255, 0)),  # 绿色
        }

        self.初始帧数: object = None
        self.显示文字: bool = False
        self.train: bool = False

        self.人脸配置: str = "assets/training_bin/opencv_face_detector.pbtxt"
        self.人脸模型: str = "assets/training_bin/opencv_face_detector_uint8.pb"
        self.年龄配置: str = "assets/training_bin/age_deploy.prototxt"
        self.年龄模型: str = "assets/training_bin/age_net.caffemodel"
        self.性别配置: str = "assets/training_bin/gender_deploy.prototxt"
        self.性别模型: str = "assets/training_bin/gender_net.caffemodel"

        self.物体检测模型名称: str = "object_detection_model"

        """
        self.object_detection_model = fasterrcnn_resnet50_fpn(pretrained=False)
        self.object_detection_model.load_state_dict(torch.load("{}.pth".format(self.物体检测模型名称)))
        self.object_detection_model.eval()
        """

    def 记录信息(self, 信息: str) -> None:
        """
        记录信息到控制台。

        Args:
            信息 (str): 要记录的信息。
        """
        现在时间: str = str(datetime.now().strftime("%H:%M:%S"))
        print(light_green + f"{self.名称} |> [{现在时间}] {信息}")

    def 计算眨眼(self, 眼睛) -> float:
        """
        计算眨眼距离。

        Args:
            眼睛 (list): 包含眼睛信息的列表。

        Returns:
            float: 眨眼距离。
        """
        if len(眼睛) == 2:
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = 眼睛

            # 计算两只眼睛的中心点
            中心_x1 = ex1 + ew1 // 2
            中心_y1 = ey1 + eh1 // 2
            中心_x2 = ex2 + ew2 // 2
            中心_y2 = ey2 + eh2 // 2

            # 计算两只眼睛之间的距离
            return np.sqrt((中心_x2 - 中心_x1) ** 2 + (中心_y2 - 中心_y1) ** 2)

    def 运行(self) -> None:
        global 左眼
        global 右眼
        global 颜色
        global 字体
        global 方向文本
        global 用户情绪
        global 性别标签

        设备名称 = self.捕获.get(cv2.CAP_PROP_POS_MSEC)

        if not os.path.exists(f"{self.物体检测模型名称}.pth"):
            _训练器_ = 训练器(文件名称=self.物体检测模型名称)
            _训练器_.运行(训练数据="assets/train", 测试数据="assets/test")
        else:
            pass

        self.物体检测模型 = torch.load(f"{self.物体检测模型名称}.pth")

        self.可用摄像头.append((self.索引, 设备名称))
        self.索引 += 1

        for self.索引, 设备名称 in self.可用摄像头:
            self.记录信息(
                信息=f"{self.捕获.__str__()} | 摄像头 {self.索引}: {设备名称}"
            )

        if not os.path.exists(self.字体路径):
            self.记录信息(信息=f"[x] {self.字体路径} 不存在! [x]")

        字体 = ImageFont.truetype(self.字体路径, 30)

        # 加载预训练模型
        人脸网络 = cv2.dnn.readNetFromTensorflow(self.人脸模型, self.人脸配置)
        性别网络 = cv2.dnn.readNetFromCaffe(self.性别配置, self.性别模型)
        年龄网络 = cv2.dnn.readNetFromCaffe(self.年龄配置, self.年龄模型)

        while True:
            # 从摄像头读取帧
            ret, f = self.捕获.read()

            帧 = cv2.flip(f, 1)

            # 将帧转换为灰度图以进行眼部检测
            灰度 = cv2.cvtColor(帧, cv2.COLOR_BGR2GRAY)

            face_region_of_interest = cv2.cvtColor(灰度, cv2.COLOR_GRAY2RGB)

            # 在帧中检测眼睛
            眼睛 = self.眼睛分类器.detectMultiScale(灰度)

            # 在帧中检测人脸
            脸部 = self.眼部分类器.detectMultiScale(灰度)

            颜色 = (255, 50, 0)

            for fx, fy, fw, fh in 脸部:
                # 绘制圆形区域以表示脸部 | 绘制具有圆角的矩形
                半径 = int(min(fw, fh) / 2)

                cv2.rectangle(
                    帧,
                    (int(fx), int(fy)),
                    (int(fx + fw), int(fy + fh)),
                    颜色,
                    thickness=2,
                )

                # 为人脸检测准备输入图像
                图像blob = cv2.dnn.blobFromImage(
                    帧, 1.0, (300, 300), [104, 117, 123], False, False
                )

                # 执行人脸检测
                人脸网络.setInput(图像blob)
                检测结果 = 人脸网络.forward()

                # 遍历检测到的人脸
                for i in range(检测结果.shape[2]):
                    置信度 = 检测结果[0, 0, i, 2]

                    # 根据置信度阈值过滤掉弱检测结果
                    if 置信度 > 0.5:
                        # 提取边界框坐标
                        边界框 = 检测结果[0, 0, i, 3:7] * np.array(
                            [帧.shape[1], 帧.shape[0], 帧.shape[1], 帧.shape[0]]
                        )

                        (起始X, 起始Y, 结束X, 结束Y) = 边界框.astype("int")

                        # 提取人脸区域
                        人脸区域 = 帧[起始Y:结束Y, 起始X:结束X]

                        # 为年龄分类准备人脸图像
                        人脸blob = cv2.dnn.blobFromImage(
                            人脸区域,
                            1.0,
                            (227, 227),
                            (78.4263377603, 87.7689143744, 114.895847746),
                            swapRB=False,
                        )

                        # 执行年龄分类
                        年龄网络.setInput(人脸blob)
                        年龄预测 = 年龄网络.forward()

                        # 为性别分类准备人脸图像
                        人脸blob = cv2.dnn.blobFromImage(
                            人脸区域,
                            1.0,
                            (227, 227),
                            (78.4263377603, 87.7689143744, 114.895847746),
                            swapRB=False,
                        )

                        # 执行性别分类
                        性别网络.setInput(人脸blob)
                        性别预测 = 性别网络.forward()

                        性别 = "Male" if 性别预测[0][0] > 性别预测[0][1] else "Female"

                        # 在帧上叠加性别标签
                        性别标签 = "{}: {:.2f}%".format(
                            性别, max(性别预测[0][0], 性别预测[0][1]) * 100
                        )

                        cv2.putText(
                            帧,
                            f"Gender: {性别标签}",
                            (起始X, 起始Y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            颜色,
                            2,
                            cv2.LINE_AA,
                            False,
                        )

                        cv2.putText(
                            帧,
                            f"Identity : Unknown",
                            (起始X, 起始Y - 90),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1,
                            颜色,
                            2,
                            cv2.LINE_AA,
                            False,
                        )
                    else:
                        continue

            """

            # Convert frame to the format expected by the object detection model
            input_image = transforms.ToTensor()(帧).unsqueeze(0)

            # Perform object detection
            with torch.no_grad():
                predictions = self.object_detection_model(input_image)

            # Process the predictions and draw bounding boxes on the frame
            for box in predictions[0]['boxes']:
                box = [int(coord) for coord in box]
                cv2.rectangle(帧, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            cv2.imshow(self.名称, 帧)

            """

            # 遍历每个检测到的眼睛
            for ex, ey, ew, eh in 眼睛:
                # 计算眼睛面积
                眼睛面积 = ew * eh

                # 检查检测到的对象是否是人眼，基于面积
                if 眼睛面积 > 1000 and 眼睛面积 < 5000:

                    # 获取眼睛中心点坐标
                    眼睛中心_x = ex + ew // 2

                    # 计算眼睛中心
                    眼睛中心 = (ex + ew // 2, ey + eh // 2)

                    # 计算圆形半径
                    半径 = max(ew, eh) // 2

                    # 根据眼睛中心点位置判断朝向
                    if self.显示文字:

                        if 眼睛中心_x < 帧.shape[1] // 3:
                            方向文本 = "向左看"  # 如果眼睛中心在帧的左侧
                            颜色 = (255, 0, 0)  # 红色

                        elif 眼睛中心_x > 2 * 帧.shape[1] // 3:
                            方向文本 = "向右看"  # 如果眼睛中心在帧的右侧
                            颜色 = (255, 0, 0)  # 蓝色

                        else:
                            方向文本 = "正视"  # 如果眼睛中心在帧的中间部分
                            颜色 = (0, 255, 0)  # 绿色

                    else:
                        方向文本 = ""  # default

                    # 用户情绪《在人脸感兴趣区域执行情绪分析》
                    结果 = DeepFace.analyze(
                        face_region_of_interest,
                        actions=["emotion"],
                        enforce_detection=self.enforce_detection,
                    )

                    # 确定主要情绪
                    情绪: str = 结果[0]["dominant_emotion"]
                    情绪准确性: float = 结果[0]["face_confidence"]

                    xRegion: float = 结果[0]["region"]["x"]
                    yRegion: float = 结果[0]["region"]["y"]
                    wRegion: float = 结果[0]["region"]["w"]
                    hRegion: float = 结果[0]["region"]["h"]

                    lEyes: list = 结果[0]["region"]["left_eye"]
                    rEyes: list = 结果[0]["region"]["right_eye"]

                    if lEyes is None and rEyes is None or xRegion == 0 and yRegion == 0:
                        print(pure_red + "[警告：请保持注意力集中在前方!!!] ")
                        self.warning(
                            帧,
                            "WARNING: PLEASE KEEP YOU EYE ON THE FRONT !!!",
                            1,
                            (0, 0, 255),
                            2,
                        )

                    # 在眼睛周围绘制一个矩形
                    # cv2.rectangle(帧, (ex, ey), (ex + ew, ey + eh), 颜色, thickness=3)

                    # 绘制圆形
                    cv2.circle(帧, 眼睛中心, 半径, 颜色, thickness=2)

                    # 在图像上绘制方向文本
                    帧_pil = Image.fromarray(cv2.cvtColor(帧, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(帧_pil)
                    draw.text((ex, ey - 30), 方向文本, font=字体, fill=颜色)
                    帧 = cv2.cvtColor(np.array(帧_pil), cv2.COLOR_RGB2BGR)

                    self.记录信息(
                        信息=f"性别: ({性别标签}) | {情绪}\t{情绪准确性 * 100} % | [ 左眼: {lEyes}, 右眼: {rEyes} ] x: {xRegion}, y: {yRegion}, w: {wRegion}, h: {hRegion}".upper()
                    )

                    cv2.putText(
                        帧,
                        str(情绪 + f" {情绪准确性 * 100} %").upper(),
                        (fx, fy - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.9,
                        颜色,
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        帧,
                        "YanJiAI",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        颜色,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        帧,
                        str(f"L: {lEyes}"),
                        (10, 55),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        颜色,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        帧,
                        str(f"R: {rEyes}"),
                        (10, 75),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        颜色,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        帧,
                        str(f"X: {xRegion}"),
                        (10, 95),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        颜色,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        帧,
                        str(f"Y: {yRegion}"),
                        (10, 115),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        颜色,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        帧,
                        str(f"W: {wRegion}"),
                        (10, 135),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        颜色,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        帧,
                        str(f"H: {hRegion}"),
                        (10, 155),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        颜色,
                        2,
                        cv2.LINE_AA,
                    )

                # 计算眨眼
                if len(眼睛) == 2:
                    眨眼距离 = self.计算眨眼(眼睛)

                    if self.初始帧数 is None:
                        self.初始帧数 = 眨眼距离

                    else:
                        if 眨眼距离 < self.初始帧数 * 0.8:

                            帧_pil = Image.fromarray(
                                cv2.cvtColor(帧, cv2.COLOR_BGR2RGB)
                            )

                            draw = ImageDraw.Draw(帧_pil)
                            帧 = cv2.cvtColor(np.array(帧_pil), cv2.COLOR_RGB2BGR)

                            self.记录信息(信息="眨眼")

                    if self.显示文字:
                        self.记录信息(信息=方向文本)

            # 显示帧
            cv2.imshow(self.名称, 帧)

            # 如果按下 'q' 键，则退出循环
            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("z")]:
                exit(0)
                break

        # 释放视频捕获
        self.捕获.release()
        cv2.destroyAllWindows()

    def warning(self, frame, text, font_scale, color, thickness):
        # 定义字体
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        # 获取文本大小
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # 计算文本在帧底部的位置
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = frame.shape[0] - 20  # 根据需要调整此值以设置距离帧底部的距离

        # 将文本放置在帧底部
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    yanjiai: object = 眼迹AI()
    yanjiai.运行()
