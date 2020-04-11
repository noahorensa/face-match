import os
import tensorflow as tf
import vggface2
import model

DATASET_PATH = "/path/to/dataset"
IMAGE_SHAPE = (130, 130, 3)
BATCH_SIZE = 8
NUM_EPOCHS = 10


def main():
  # prepare the dataset
  num_examples, train_ds = vggface2.preprocess_dataset(DATASET_PATH, IMAGE_SHAPE)
  print(train_ds)
  print('Dataset preparation complete. NUM_EXAMPLES =', num_examples)
  train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).repeat()

  STEPS_PER_EPOCH = int(num_examples / BATCH_SIZE)

  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="vggface2.ckpt", save_weights_only=True, verbose=1
  )

  csv_callback = tf.keras.callbacks.CSVLogger(
    "vggface2.csv", separator=',', append=False
  )

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
      callbacks=[ckpt_callback, csv_callback],
      verbose=1
    )


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  main()
