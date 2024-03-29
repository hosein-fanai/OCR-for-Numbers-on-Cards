from matplotlib import pyplot as plt


def plot_history(history):
    epochs = range(1, len(history["loss"])+1)

    train_loss = history["loss"]
    val_loss = history["val_loss"]

    train_confs_acc = history["confs_accuracy"]
    val_confs_acc = history["val_confs_accuracy"]

    train_classes_acc = history["classes_accuracy"]
    val_classes_acc = history["val_classes_accuracy"]

    train_bboxes_mae = history["bboxes_loss"]
    val_bboxes_mae = history["val_bboxes_loss"]

    train_card_type_acc = history.get("card_type_accuracy", [0]*len(history["loss"]))
    val_card_type_acc = history.get("val_card_type_accuracy", [0]*len(history["loss"]))

    train_cvv2_acc = history.get("cvv2_accuracy", [0]*len(history["loss"]))
    val_cvv2_acc = history.get("val_cvv2_accuracy", [0]*len(history["loss"]))

    train_exp_date_acc = history.get("exp_date_accuracy", [0]*len(history["loss"]))
    val_exp_date_acc = history.get("val_exp_date_accuracy", [0]*len(history["loss"]))

    plt.figure(figsize=(15, 15))

    plt.subplot(4, 2, 1)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(epochs, train_card_type_acc, label="Train")
    plt.plot(epochs, val_card_type_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Card Type Accuracy")
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(epochs, train_cvv2_acc, label="Train")
    plt.plot(epochs, val_cvv2_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("CVV2 Accuracy")
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(epochs, train_exp_date_acc, label="Train")
    plt.plot(epochs, val_exp_date_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Expiration Date Accuracy")
    plt.legend()

    plt.subplot(4, 2, 5)
    plt.plot(epochs, train_confs_acc, label="Train")
    plt.plot(epochs, val_confs_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Confs Accuracy")
    plt.legend()

    plt.subplot(4, 2, 6)
    plt.plot(epochs, train_bboxes_mae, label="Train")
    plt.plot(epochs, val_bboxes_mae, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BBoxes MAE")
    plt.legend()

    plt.subplot(4, 2, 7)
    plt.plot(epochs, train_classes_acc, label="Train")
    plt.plot(epochs, val_classes_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Classes Accuracy")
    plt.legend()

    plt.show()


