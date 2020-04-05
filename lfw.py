import tensorflow as tf
import tensorflow_datasets as tfds

NUM_EXAMPLES = 13233
NUM_CLASSES = 5749

def preprocess_dataset(image_size=(72, 72)):
  train_ds = tfds.load(name="lfw", split=tfds.Split.TRAIN)

  labels = train_ds.map(lambda example: example['label'])
  label_dict = {}
  unique_labels = []
  label_id = 0

  for l in labels:
    s = l.numpy()
    if s not in label_dict:
      label_dict[s] = label_id
      unique_labels.append(s)
      label_id += 1

  encoder = tfds.features.text.TokenTextEncoder(unique_labels)

  def format_example(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, image_size)
    label = encoder.encode(label.numpy())
    return image, label

  @tf.function
  def tf_format_example(example):
    return tf.py_function(
      format_example,
      [example['image'], example['label']],
      [tf.float32, tf.uint64]
    )

  train_ds = train_ds.map(tf_format_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return train_ds


def write_dataset(ds, filename="lfw.tfrecord"):
  def serialize_example(image, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
      'image': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

  @tf.function
  def tf_serialize_example(image, label):
    tf_string = tf.py_function(
      serialize_example,
      (tf.reshape(image, [-1]), label),  # pass these args to the above function.
      tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar

  serialized_dataset = ds.map(tf_serialize_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(serialized_dataset)

  return


def read_dataset(filename="lfw.tfrecord"):
  feature_description = {
    'image': tf.io.FixedLenFeature(shape=[72, 72, 3], dtype=tf.float32),
    'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
  }

  def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example['image'], example['label']

  filenames = [filename]
  ds = tf.data.TFRecordDataset(filenames)
  ds = ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds


def main():
  ds = preprocess_dataset()
  write_dataset(ds)


if __name__ == "__main__":
  main()
