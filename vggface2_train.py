import tensorflow as tf
import vggface2
import model

DATASET_PATH = "/path/to/dataset"
IMAGE_SHAPE = (96, 96, 3)
BATCH_SIZE = 10
NUM_EPOCHS = 20
STEPS_PER_EPOCH = int(vggface2.NUM_EXAMPLES / BATCH_SIZE)


def main():
  # prepare the dataset
  train_ds = vggface2.preprocess_dataset(DATASET_PATH)
  print(train_ds)
  train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).repeat()

  with tf.device('/cpu:0'):
    # prepare the model
    m = model.create_model(
      shape=IMAGE_SHAPE,
      num_classes=vggface2.NUM_CLASSES
    )

    # train the model
    m.fit(
      train_ds,
      epochs=NUM_EPOCHS,
      steps_per_epoch=STEPS_PER_EPOCH,
      verbose=1
    )


if __name__ == "__main__":
  main()
