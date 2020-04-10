import os
import json
import tensorflow as tf
import vggface2
import model

WORKERS = []
WORKER_INDEX = 0
NUM_WORKERS = len(WORKERS)

DATASET_PATH = "/path/to/dataset"
IMAGE_SHAPE = (96, 96, 3)
BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
NUM_EPOCHS = 10
STEPS_PER_EPOCH = int(vggface2.NUM_EXAMPLES / BATCH_SIZE)


def main():

  dist_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

  # prepare the dataset
  train_ds = vggface2.preprocess_dataset(DATASET_PATH)
  print(train_ds)
  train_ds = train_ds\
    .with_options(options)\
    .shard(NUM_WORKERS, WORKER_INDEX)\
    .batch(BATCH_SIZE, drop_remainder=True)\
    .repeat()

  with dist_strategy.scope():
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

  os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
      'worker': WORKERS
    },
    'task': {'type': 'worker', 'index': WORKER_INDEX}
  })

  main()
