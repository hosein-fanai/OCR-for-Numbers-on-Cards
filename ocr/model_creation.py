import tensorflow as tf 
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras import backend as K

from ocr_model import OCRModel

from constants import (input_shape, dropout_rate, num_anchors, num_classes, 
                    model_name, lr, reg_coef, model_path, train_with_masks, 
                    training_phase_2, lr_phase_2)


def load_model(model_path=model_path):
    return models.load_model(model_path, custom_objects={"OCRModel": OCRModel})


def create_sliding_window_ocr_model(prev_model_path=None, layers_to_freeze=[]):
    K.clear_session()


    conv_base = applications.Xception(include_top=False, input_shape=input_shape)
    for layer in conv_base.layers[:-15]:
        layer.trainable = False

    input_image = layers.Input(shape=input_shape, dtype=tf.uint8, name="image")

    x = models.Sequential([
        layers.RandomBrightness(0.5),
        layers.RandomContrast(0.5),
    ], name="data_augmentation")(input_image)

    x = layers.Lambda(lambda X: applications.xception.preprocess_input(X), name="xception_preprocess")(x)

    x = conv_base(x)


    if training_phase_2:
        z = layers.GlobalAveragePooling2D()(x)
        z = layers.Dropout(dropout_rate)(z)

        z1 = layers.Dense(1024, activation="relu", name="bottleneck_cvv2_digits_dense_1")(z)
        z1 = layers.Dropout(dropout_rate, name="bottleneck_digits_do_1")(z1)

        z1 = layers.Dense(512, activation="relu", name="bottleneck_cvv2_digits_dense_2")(z1)
        z1 = layers.Dropout(dropout_rate, name="bottleneck_digits_do_2")(z1)

        z2 = layers.Dense(1024, activation="relu", name="bottleneck_exp_date_digits_dense_1")(z)
        z2 = layers.Dropout(dropout_rate, name="bottleneck_digits_do_1")(z2)

        z2 = layers.Dense(512, activation="relu", name="bottleneck_exp_date_digits_dense_2")(z2)
        z2 = layers.Dropout(dropout_rate, name="bottleneck_digits_do_2")(z2)

        card_type = layers.Dense(1, activation="sigmoid", name="card_type")(z)
        cvv2_outputs = [layers.Dense(10, activation="softmax", name=f"cvv2_digit_{i}")(z1) for i in range(4)]
        exp_date_outputs = [layers.Dense(10, activation="softmax", name=f"exp_date_digit_{i}")(z2) for i in range(8)]


    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    x1 = layers.SeparableConv2D(512, kernel_size=1, padding="same", use_bias=False, name="bboxes_anchors_bottleneck_conv_1")(x)
    x1 = layers.BatchNormalization(name="bboxes_anchors_bottleneck_bn_1")(x1)
    x1 = layers.ReLU(name="bboxes_anchors_bottleneck_relu_1")(x1)

    x1 = layers.SeparableConv2D(256, kernel_size=1, padding="same", use_bias=False, name="bboxes_anchors_bottleneck_conv_2")(x1)
    x1 = layers.BatchNormalization(name="bboxes_anchors_bottleneck_bn_2")(x1)
    x1 = layers.ReLU(name="bboxes_anchors_bottleneck_relu_2")(x1)

    x1 = layers.SeparableConv2D(128, kernel_size=1, padding="same", use_bias=False, name="bboxes_anchors_bottleneck_conv_3")(x1)
    x1 = layers.BatchNormalization(name="bboxes_anchors_bottleneck_bn_3")(x1)
    x1 = layers.ReLU(name="bboxes_anchors_bottleneck_relu_3")(x1)

    confs_outputs = [layers.SeparableConv2D(1, kernel_size=1, padding="same", activation="sigmoid", name=f"confs_anchor_{i}")(x) for i in range(num_anchors)]
    bboxes_outputs = [layers.SeparableConv2D(4, kernel_size=1, padding="same", activation="linear", name=f"bboxes_anchor_{i}")(x1) for i in range(num_anchors)]
    classes_outputs = [layers.SeparableConv2D(num_classes, kernel_size=1, padding="same", activation="softmax", name=f"classes_anchor_{i}")(x) for i in range(num_anchors)]


    if train_with_masks:
        model_type = OCRModel
    else:
        model_type = models.Model

    model = model_type(
        inputs=input_image, 
        outputs={
            **{f"confs_anchor_{i}": confs_outputs[i] for i in range(num_anchors)}, 
            **{f"bboxes_anchor_{i}": bboxes_outputs[i] for i in range(num_anchors)},
            **{f"classes_anchor_{i}": classes_outputs[i] for i in range(num_anchors)},
        } | ({"card_type": card_type} if training_phase_2 else {}) | \
            ({f"cvv2_digit_{i}": cvv2_outputs[i] for i in range(4)} if training_phase_2 else {}) | \
            ({f"exp_date_digit_{i}": exp_date_outputs[i] for i in range(8)} if training_phase_2 else {}), 
        name=model_name
    )

    if training_phase_2:
        assert prev_model_path, "Didn't get the path to the previous model (the model for phase 1)."

        learning_rate = lr_phase_2

        prev_model = load_model(prev_model_path)
        for layer in model.layers:
            try:
                model.get_layer(layer.name).set_weights(prev_model.get_layer(layer.name).get_weights())
            except:
                continue
    else:
        learning_rate = lr


    for layer_name in layers_to_freeze:
        model.get_layer(layer_name).trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate, decay=reg_coef), 
        loss={
            **{f"confs_anchor_{i}": "binary_crossentropy" for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": "mae" for i in range(num_anchors)},
            **{f"classes_anchor_{i}": "sparse_categorical_crossentropy" for i in range(num_anchors)},
        } | ({"card_type": "binary_crossentropy"} if training_phase_2 else {}) | \
            ({f"cvv2_digit_{i}": "sparse_categorical_crossentropy" for i in range(4)} if training_phase_2 else {}) | \
            ({f"exp_date_digit_{i}": "sparse_categorical_crossentropy" for i in range(8)} if training_phase_2 else {}),
        metrics={
            **{f"confs_anchor_{i}": ["accuracy"] for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": ["mse"] for i in range(num_anchors)},
            **{f"classes_anchor_{i}": ["accuracy"] for i in range(num_anchors)},
        } | ({"card_type": ["accuracy"]} if training_phase_2 else {}) | \
            ({f"cvv2_digit_{i}": ["accuracy"] for i in range(4)} if training_phase_2 else {}) | \
            ({f"exp_date_digit_{i}": ["accuracy"] for i in range(8)} if training_phase_2 else {}), 
        loss_weights={
            **{f"confs_anchor_{i}": 1. for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": 1. for i in range(num_anchors)},
            **{f"classes_anchor_{i}": 1. for i in range(num_anchors)}, 
        } | ({"card_type": 0.1} if training_phase_2 else {} if training_phase_2 else {}) | \
            ({f"cvv2_digit_{i}": 0.1 for i in range(4)} if training_phase_2 else {}) | \
            ({f"exp_date_digit_{i}": 0.1 for i in range(8)} if training_phase_2 else {}), 
    )


    return model


