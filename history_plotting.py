from matplotlib import pyplot as plt


def plot_history(history):
    epochs = range(1, len(history["loss"])+1)

    train_loss = history["loss"]
    val_loss = history["val_loss"]

    train_confs_acc = history["confs_accuracy"]
    val_confs_acc = history["val_confs_accuracy"]

    train_classes_acc = history["classes_accuracy"]
    val_classes_acc = history["val_classes_accuracy"]

    train_bboxes_mae = history["bboxes_mae"]
    val_bboxes_mae = history["val_bboxes_mae"]

    train_card_type_acc = history["card_type_accuracy"]
    val_card_type_acc = history["val_card_type_accuracy"]

    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")

    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_card_type_acc, label="Train")
    plt.plot(epochs, val_card_type_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Card Type Accuracy")

    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_confs_acc, label="Train")
    plt.plot(epochs, val_confs_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Confs Accuracy")

    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_bboxes_mae, label="Train")
    plt.plot(epochs, val_bboxes_mae, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BBoxes MAE")

    plt.subplot(3, 2, 5)
    plt.plot(epochs, train_classes_acc, label="Train")
    plt.plot(epochs, val_classes_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Classes Accuracy")

    plt.legend()
    plt.show()


