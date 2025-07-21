import tensorflow as tf
from tensorflow import keras
from keras import layers

def load_and_prepare_dataset(
    data_path,
    image_size=(224, 224),
    batch_size=32,
    val_split=0.2,
    seed=123,
):
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=val_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    # prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds