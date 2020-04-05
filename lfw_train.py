IMAGE_SHAPE = (72, 72, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 20


def main():
  import lfw

  # prepare the model
  from model import create_model
  model = create_model(
    shape=IMAGE_SHAPE,
    num_classes=lfw.NUM_CLASSES
  )

  # prepare the dataset
  train_ds = lfw.read_dataset()
  print(train_ds)
  train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).repeat()

  # train the model
  model.fit(
    train_ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=lfw.NUM_EXAMPLES / BATCH_SIZE
  )


if __name__ == "__main__":
  main()
