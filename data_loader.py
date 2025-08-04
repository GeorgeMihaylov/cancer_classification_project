import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Путь к папке с данными
DATA_DIR = 'data/IDC_regular_ps50_idx5/'

# Параметры изображений
IMG_HEIGHT = 50
IMG_WIDTH = 50
BATCH_SIZE = 32


# Генератор для аугментации и загрузки данных
def create_data_generators():
    """
    Создает и возвращает генераторы данных для обучения, валидации и тестирования.
    Применяет аугментацию для тренировочной выборки и масштабирование для всех выборок.
    """

    # 1. Сбор путей ко всем изображениям
    all_image_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.png'):
                all_image_paths.append(os.path.join(root, file))

    # 2. Разделение данных на тренировочную, валидационную и тестовую выборки
    # Для этого датасета нет готового разделения, поэтому мы сделаем его сами.
    # Мы будем использовать 80% данных для обучения и 20% для валидации/тестирования.

    from sklearn.model_selection import train_test_split

    # Создаем фиктивные метки, так как train_test_split требует их наличия
    all_labels = [os.path.basename(os.path.dirname(p)) for p in all_image_paths]

    # Делим данные на тренировочную и оставшуюся часть
    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, stratify=all_labels
    )

    # Делим оставшуюся часть на валидационную и тестовую
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        remaining_paths, remaining_labels, test_size=0.5, stratify=remaining_labels
    )

    # 3. Создание ImageDataGenerator с аугментацией
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Генераторы для валидации и тестирования без аугментации, только с масштабированием
    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    # 4. Создание и возвращение генераторов
    # Мы будем использовать flow_from_dataframe, так как у нас нет готовой структуры папок
    # для train_val_test.

    import pandas as pd

    train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels})
    val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels})
    test_df = pd.DataFrame({'filename': test_paths, 'class': test_labels})

    train_data_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_data_gen = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_data_gen = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_data_gen, val_data_gen, test_data_gen


# Проверка загрузки данных (необязательно, но полезно)
if __name__ == '__main__':
    print("Загрузка данных...")
    train_gen, val_gen, test_gen = create_data_generators()

    print(f"Количество изображений для обучения: {train_gen.n}")
    print(f"Количество изображений для валидации: {val_gen.n}")
    print(f"Количество изображений для тестирования: {test_gen.n}")
    print(f"Названия классов: {list(train_gen.class_indices.keys())}")

    print("\nПроцесс загрузки завершен.")