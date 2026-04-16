import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Input
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD, Adam


# НАЛАШТУВАННЯ

np.random.seed(42)

SHOW_SAMPLES = True
TRAIN_MNIST = True
TRAIN_CIFAR10 = True


# для MNIST бажано чорно-біла цифра 28x28
MNIST_CUSTOM_IMAGES = [
    "num2.png",
     "num7.png",
     "num0.png",
     "num3.png",
     "num4.png",
     "num.png",
]

# для CIFAR-10 бажано кольорове зображення 32x32
CIFAR_CUSTOM_IMAGES = [
    "Screenshot_3.png",
    # "my_car.png",
]

CIFAR_CLASSES = [
    "літак",
    "автомобіль",
    "птах",
    "кіт",
    "олень",
    "собака",
    "жаба",
    "кінь",
    "корабель",
    "вантажівка"
]


# ДОПОМІЖНІ ФУНКЦІЇ
def show_mnist_examples(x, y, count=10):
    plt.figure(figsize=(12, 3))
    for i in range(count):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i], cmap="gray")
        plt.title(f"Цифра: {y[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_cifar_examples(x, y, count=10):
    plt.figure(figsize=(14, 3))
    for i in range(count):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i].astype("uint8"))
        plt.title(CIFAR_CLASSES[int(y[i][0])])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ЧАСТИНА 1. MLP ДЛЯ MNIST
def load_and_prepare_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if SHOW_SAMPLES:
        show_mnist_examples(x_train, y_train)

    # reshape: 28x28 -> 784
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # normalization
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # one-hot encoding
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return x_train, y_train_cat, x_test, y_test_cat


def build_mnist_mlp_basic():
    """
    Базова модель з методички:
    Dense(800) -> Dense(10)
    """
    model = Sequential([
        Dense(800, input_dim=784, activation="relu", kernel_initializer="normal"),
        Dense(10, activation="softmax", kernel_initializer="normal")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(),
        metrics=["accuracy"]
    )
    return model


def build_mnist_mlp_improved():
    """
    Покращена модель для завдання 1.2:
    2 приховані шари + dropout.
    Дає вищу точність.
    """
    model = Sequential([
        Dense(512, input_dim=784, activation="relu"),
        Dropout(0.2),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    return model


def train_mnist_models():
    print_separator("ЧАСТИНА 1. РОЗПІЗНАВАННЯ MNIST ЗА ДОПОМОГОЮ MLP")

    x_train, y_train, x_test, y_test = load_and_prepare_mnist()

    # БАЗОВА МОДЕЛЬ
    print("\n[1] Базова MLP-модель")
    basic_model = build_mnist_mlp_basic()
    basic_model.summary()

    basic_model.fit(
        x_train,
        y_train,
        batch_size=200,
        epochs=20,
        validation_split=0.2,
        verbose=2
    )

    basic_scores = basic_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nТочність базової MLP на тестових даних: {basic_scores[1] * 100:.2f}%")

    # збереження
    basic_model.save("mnist_mlp_basic.keras")
    print("Модель збережено у файл: mnist_mlp_basic.keras")

    # ПОКРАЩЕНА МОДЕЛЬ
    print("\n[2] Покращена MLP-модель")
    improved_model = build_mnist_mlp_improved()
    improved_model.summary()

    improved_model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=20,
        validation_split=0.2,
        verbose=2
    )

    improved_scores = improved_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nТочність покращеної MLP на тестових даних: {improved_scores[1] * 100:.2f}%")

    improved_model.save("mnist_mlp_improved.keras")
    print("Модель збережено у файл: mnist_mlp_improved.keras")

    return improved_model


def predict_mnist_custom_images(model, image_paths):
    if not image_paths:
        return

    print_separator("ПЕРЕВІРКА MLP НА ВЛАСНИХ ЗОБРАЖЕННЯХ")

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Файл не знайдено: {img_path}")
            continue

        # grayscale + 28x28
        img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
        img_array = image.img_to_array(img)

        # нормалізація
        img_array = img_array.astype("float32") / 255.0

        img_array = 1.0 - img_array

        plt.imshow(img_array.squeeze(), cmap="gray")
        plt.title(f"Перевірка: {os.path.basename(img_path)}")
        plt.axis("off")
        plt.show()

        img_array = img_array.reshape(1, 784)
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)

        print(f"{img_path} -> передбачена цифра: {predicted_digit}")


# ЧАСТИНА 2. CNN ДЛЯ CIFAR-10
def load_and_prepare_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if SHOW_SAMPLES:
        show_cifar_examples(x_train, y_train)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    num_classes = 10
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    return x_train, y_train_cat, x_test, y_test_cat


def build_cifar_cnn_basic():
    """
    CNN по мотивам методички:
    Conv -> Conv -> Pool -> Dropout
    Conv -> Conv -> Pool -> Dropout
    Flatten -> Dense -> Dropout -> Dense
    """
    input_shape = (32, 32, 3)

    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    out = Dense(10, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    return model


def build_cifar_cnn_improved():
    """
    Трохи покращена версія для експериментів.
    """
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    return model


def train_cifar_models():
    print_separator("ЧАСТИНА 2. КЛАСИФІКАЦІЯ CIFAR-10 ЗА ДОПОМОГОЮ CNN")

    x_train, y_train, x_test, y_test = load_and_prepare_cifar10()

    # БАЗОВА МОДЕЛЬ
    print("\n[1] Базова CNN-модель")
    basic_model = build_cifar_cnn_basic()
    basic_model.summary()

    basic_model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.1,
        verbose=2
    )

    basic_scores = basic_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nТочність базової CNN на тестових даних: {basic_scores[1] * 100:.2f}%")

    basic_model.save("cifar10_cnn_basic.keras")
    print("Модель збережено у файл: cifar10_cnn_basic.keras")

    # ПОКРАЩЕНА МОДЕЛЬ
    print("\n[2] Покращена CNN-модель")
    improved_model = build_cifar_cnn_improved()
    improved_model.summary()

    improved_model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=12,
        validation_split=0.1,
        verbose=2
    )

    improved_scores = improved_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nТочність покращеної CNN на тестових даних: {improved_scores[1] * 100:.2f}%")

    improved_model.save("cifar10_cnn_improved.keras")
    print("Модель збережено у файл: cifar10_cnn_improved.keras")

    return improved_model


def predict_cifar_custom_images(model, image_paths):
    if not image_paths:
        return

    print_separator("ПЕРЕВІРКА CNN НА ВЛАСНИХ ЗОБРАЖЕННЯХ")

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Файл не знайдено: {img_path}")
            continue

        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)

        plt.imshow(img)
        plt.title(f"Перевірка: {os.path.basename(img_path)}")
        plt.axis("off")
        plt.show()

        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)

        print(f"{img_path} -> клас: {CIFAR_CLASSES[predicted_class]}")
        print(f"Ймовірності: {prediction[0]}")
        print("-" * 50)


# MAIN
if __name__ == "__main__":
    mnist_model = None
    cifar_model = None

    if TRAIN_MNIST:
        mnist_model = train_mnist_models()
        predict_mnist_custom_images(mnist_model, MNIST_CUSTOM_IMAGES)

    if TRAIN_CIFAR10:
        cifar_model = train_cifar_models()
        predict_cifar_custom_images(cifar_model, CIFAR_CUSTOM_IMAGES)

    print_separator("РОБОТУ ЗАВЕРШЕНО")
    print("Усі етапи лабораторної роботи виконані.")