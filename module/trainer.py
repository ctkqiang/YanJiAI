# -*- coding: UTF-8 -*-
try:
    import os
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
except ImportError:
    raise ImportError("🥹无法安装配件")
finally:
    pass

class Trainer: 
    def __init__(self, 文件名称: str) -> None:
        super(Trainer, self).__init__()

        self.文件名称: str = 文件名称
        
        assert tf.__version__.startswith("2")

        self.fill_mode: str = "nearest"

    def 运行(self, 数据: str, 测试数据: str) -> None:

        if not os.path.exists(数据):
            raise "a"

        if not os.path.exists(测试数据):
            raise "b"


        # 输出媒体数据
        媒体数据 = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode=self.fill_mode
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = 媒体数据.flow_from_directory(
            数据,
            target_size=(150, 150),
            batch_size=32,
            class_mode="binary"  # use "categorical" for multi-class classification <== 
        )

        val_generator = val_datagen.flow_from_directory(
            测试数据,
            target_size=(150, 150),
            batch_size=32,
            class_mode="binary"
        )

        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")  # use "softmax" for multi-class classification
        ])

        model.summary()




if __name__ == "__main__":
    trainer = Trainer(文件名称="object_model.pb")
    trainer.运行(数据="assets/train", 测试数据="assets/test")
