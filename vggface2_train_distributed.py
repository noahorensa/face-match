import os
import json
import tensorflow as tf
import vggface2
import model
import subprocess

out = subprocess.Popen(['hostname'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
hostname = out.communicate()

WORKERS = ["discus-spark1:12345", "discus-spark2:12345", "discus-spark3:12345", "discus-spark4:12345", "discus-spark5:12345", "discus-spark6:12345", "discus-spark7:12345", "discus-spark8:12345", "discus-spark9:12345"]
WORKER_INDEX = int(hostname[0].decode('UTF-8')[-2]) - 1
NUM_WORKERS = len(WORKERS)

DATASET_PATH = "/trux/data/VGGFace2/train"
IMAGE_SHAPE = (96, 96, 3)
BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
NUM_EPOCHS = 10
STEPS_PER_EPOCH = int(vggface2.NUM_EXAMPLES / GLOBAL_BATCH_SIZE)

print("WORKER =", hostname)
print("WORKER_INDEX =", WORKER_INDEX)
print("DATASET_PATH =", DATASET_PATH)
print("BATCH_SIZE =", BATCH_SIZE)
print("GLOBAL_BATCH_SIZE =", GLOBAL_BATCH_SIZE)


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
    .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)\
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
        num_classes=vggface2.NUM_CLASSES + 1
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
