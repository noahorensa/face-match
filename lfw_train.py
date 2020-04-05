import tensorflow as tf
import lfw
import model

IMAGE_SHAPE = (72, 72, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 20


def main():
  # prepare the dataset
  train_ds = lfw.read_dataset()
  print(train_ds)
  train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).repeat()

  with tf.device('/gpu:0'):
    # prepare the model
    m = model.create_model(
      shape=IMAGE_SHAPE,
      num_classes=lfw.NUM_CLASSES
    )

    # train the model
    m.fit(
      train_ds,
      epochs=NUM_EPOCHS,
      steps_per_epoch=lfw.NUM_EXAMPLES / BATCH_SIZE
    )


if __name__ == "__main__":
  main()
