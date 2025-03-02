import tensorflow as tf
import tensorflow_datasets as tfds
import os

NUM_EXAMPLES = 3141890
NUM_CLASSES = 1000 + 1
# NUM_CLASSES = 8631 + 1

def preprocess_dataset(path, image_size=(96, 96, 3)):
  print("Listing directories in ", path)

  label_dict = {}
  unique_labels = []
  label_id = 0

  for root, dirs, files in os.walk(path):
    for d in dirs:
      l = tf.strings.split(d, os.path.sep)[-1].numpy()
      label_dict[l.decode('UTF-8')] = label_id
      unique_labels.append(l)
      label_id += 1
      if (len(unique_labels) == NUM_CLASSES - 1):
        break

  def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2]

  encoder = tfds.features.text.TokenTextEncoder(unique_labels)

  def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size.
    img = tf.image.resize_with_pad(img, image_size[0], image_size[1])
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # get label
    label = encoder.encode(get_label(file_path).numpy())
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

  @tf.function
  def tf_process_path(path):
    image, label = tf.py_function(
      process_path,
      [path],
      [tf.float32, tf.float32]
    )
    image.set_shape((image_size[0], image_size[1], 3))
    label.set_shape((1, NUM_CLASSES))
    return image, label

  try:
    f = open("vggface2.lst")

    print("Reading vggface2.lst")
    files = []
    for x in f.readlines():
      file = x[0:-1]
      if file.split(os.path.sep)[-2] in label_dict:
        files.append(file)
    ds_len = len(files)
    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(files, dtype=tf.string))

  except FileNotFoundError:
    print("vggface2.lst not found. Listing jpg files in", path)

    ds = tf.data.Dataset.list_files(path + "/*/*.jpg").shuffle(1000)

    print("Writing vggface.lst")
    f = open("vggface2.lst", "w")
    for i in ds:
      f.write(i.numpy().decode('UTF-8') + '\n')

  finally:
    f.close()

  print("Preparing dataset...")

  ds = ds.map(tf_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds_len, ds


def write_dataset(ds, filename="vggface2.tfrecord"):
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

  def tf_serialize_example(image, label):
    tf_string = tf.py_function(
      serialize_example,
      (tf.reshape(image, [-1]), label),  # pass these args to the above function.
      tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar

  serialized_dataset = ds.map(tf_serialize_example)

  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(serialized_dataset)

  return


def read_dataset(filename="vggface2.tfrecord"):
  feature_description = {
    'image': tf.io.FixedLenFeature(shape=[96, 96, 3], dtype=tf.float32),
    'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
  }

  def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example['image'], example['label']

  filenames = [filename]
  ds = tf.data.TFRecordDataset(filenames)
  ds = ds.map(_parse_function)

  return ds


def main():
  ds = preprocess_dataset("/media/noah/Noah/VGGFace2/test")
  write_dataset(ds)


if __name__ == "__main__":
  main()
