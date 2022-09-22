import tensorflow as tf
import numpy as np

def get_data_generators(VALIDATION_SPLIT, IMAGE_SIZE, BATCH_SIZE, data_dir, DATA_AUGMENTATION, train_dir = None, val_dir = None):
  if train_dir is not None:
    datagen_kwargs = dict( rescale=1./255)
  else:
    datagen_kwargs = dict( rescale=1./255, validation_split= VALIDATION_SPLIT )

  dataflow_kwargs = dict(
        target_size=IMAGE_SIZE[:-1],
        batch_size = BATCH_SIZE,
        interpolation="bilinear")

  valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
  if train_dir is not None:
        valid_generator = valid_datagen.flow_from_directory(
        directory=val_dir,
        
        shuffle=False,
        **dataflow_kwargs)
  else:
    valid_generator = valid_datagen.flow_from_directory(
        directory=data_dir,
        subset="validation",
        shuffle=False,
        **dataflow_kwargs)

  if DATA_AUGMENTATION:
      train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rotation_range=40,
          horizontal_flip=True,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          **datagen_kwargs
      )
  else:
      train_datagen = valid_datagen
  if train_dir is not None:
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        subset="training",
        shuffle=True,
        **dataflow_kwargs)
  else:
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        subset="training",
        shuffle=True,
        **dataflow_kwargs)
  return train_generator, valid_generator


def manage_input_data(input_image, IMAGE_SIZE):
    """converting the input array into desired dimension
    Args:
        input_image (nd array): image nd array
    Returns:
        nd array: resized and updated dim image
    """
    images = input_image
    size = IMAGE_SIZE[:-1]
    resized_input_img = tf.image.resize(images, size)
    final_img = np.expand_dims(resized_input_img, axis=0)
    
    return final_img