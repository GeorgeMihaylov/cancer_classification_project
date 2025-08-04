import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16  # Импортируем VGG16
from data_loader import create_data_generators
import matplotlib.pyplot as plt
import numpy as np

# Параметры обучения
EPOCHS = 10
BATCH_SIZE = 32
IMG_HEIGHT = 50
IMG_WIDTH = 50

# Создание генераторов данных
train_data_gen, val_data_gen, test_data_gen = create_data_generators()


# Определение архитектуры модели с использованием Transfer Learning
def create_model():
    # Загрузка предобученной модели VGG16 без верхних слоев
    # include_top=False, чтобы мы могли добавить свои слои
    # input_shape - размер наших изображений
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Замораживаем слои базовой модели
    # Это предотвратит их переобучение
    base_model.trainable = False

    # Создание новой модели Sequential
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Построение и вывод структуры модели
model = create_model()
model.summary()

# Обучение модели
print("\nНачало обучения модели...")
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // BATCH_SIZE
)
print("Обучение завершено.")

# Визуализация результатов обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Оценка модели на тестовой выборке
test_loss, test_acc = model.evaluate(test_data_gen)
print(f'\nТочность на тестовой выборке: {test_acc:.4f}')

# Сохранение модели
model.save('cancer_detection_vgg16_model.h5')
print('Модель сохранена как cancer_detection_vgg16_model.h5')