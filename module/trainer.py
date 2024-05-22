# -*- coding: UTF-8 -*-
try:
    import os
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import legacy
    from PIL import Image
except ImportError:
    raise ImportError("🥹无法安装配件")
finally:
    pass

class Trainer:
    def __init__(self, 文件名称: str) -> None:
        super(Trainer, self).__init__()

        self.文件名称: str = 文件名称

        assert tf.__version__.startswith("2"), "This script requires TensorFlow 2.x."

        self.fill_mode: str = "nearest"

    def check_images(self, directory: str) -> None:
        for filename in os.listdir(directory):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(os.path.join(directory, filename))
                    img.verify()  # Verify that it is, in fact, an image
                except (IOError, SyntaxError) as e:
                    print(f"Bad file: {filename}")  # Print out the names of corrupt files

    def 运行(self, 数据: str, 测试数据: str) -> None:

        if not os.path.exists(数据):
            raise FileNotFoundError(f"训练数据目录 '{数据}' 不存在.")
        
        if not os.path.exists(测试数据):
            raise FileNotFoundError(f"测试数据目录 '{测试数据}' 不存在.")

        # Check images in directories
        self.check_images(数据)
        self.check_images(测试数据)

        # 创建 ImageDataGenerator 实例
        train_datagen = ImageDataGenerator(
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

        # 加载训练数据
        train_generator = train_datagen.flow_from_directory(
            数据,
            target_size=(150, 150),
            batch_size=32,
            class_mode="binary"  # 使用 "categorical" 进行多类分类
        )

        # 加载验证数据
        val_generator = val_datagen.flow_from_directory(
            测试数据,
            target_size=(150, 150),
            batch_size=32,
            class_mode="binary"
        )

        # 定义模型
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
            Dense(1, activation="sigmoid")  # 使用 "softmax" 进行多类分类
        ])

        model.summary()

        # 编译模型
        model.compile(
            loss="binary_crossentropy",  # 使用 "categorical_crossentropy" 进行多类分类
            optimizer=legacy.RMSprop(learning_rate=1e-4),
            metrics=["accuracy"]
        )

        # 训练模型
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=10,
            validation_data=val_generator,
            validation_steps=val_generator.samples // val_generator.batch_size
        )

        # 评估模型
        loss, accuracy = model.evaluate(val_generator)
        
        print(f'验证损失: {loss}')
        print(f'验证准确率: {accuracy}')


if __name__ == "__main__":
    trainer = Trainer(文件名称="object_model.pb")
    trainer.运行(数据="assets/train", 测试数据="assets/test")