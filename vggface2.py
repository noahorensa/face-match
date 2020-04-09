import tensorflow as tf
import os

NUM_EXAMPLES = 3141890
NUM_CLASSES = 8631

def preprocess_dataset(path, image_size=(96, 96)):
  label_dict = {}
  label_id = 0
  for root, dirs, files in os.walk(path):
    for d in dirs:
      label_dict[bytes(d, 'utf-8')] = label_id
      label_id += 1

  def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2]

  def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, image_size)

  def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    # get label
    label = tf.py_function(lambda l: label_dict[l.numpy()], [get_label(file_path)], tf.uint64)
    label.set_shape(tf.TensorShape(()))
    return img, label

  ds = tf.data.Dataset.list_files(path + "/*/*.jpg")
  ds = ds.map(process_path)

  return ds


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
