"""File to load train and test data"""

import tensorflow as tf


def load_data(
    path_to_train_test: str,
    batch_size: int = 32,
    seed: int = 42
) -> tuple:
    AUTOTUNE = tf.data.AUTOTUNE

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        f'{path_to_train_test}/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        f'{path_to_train_test}/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        f'{path_to_train_test}/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
