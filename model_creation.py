import tensorflow as tf 
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras import backend as K

from ocr_model import OCRModel
from constants import input_shape, dropout_rate, num_anchors, num_classes, model_name, lr, reg_coef, model_path


def create_sliding_window_ocr_model():
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


    # z = layers.GlobalAveragePooling2D()(x)

    # z1 = layers.Dense(1024, activation="relu", name="bottleneck_digits_dense_1")(z)
    # z1 = layers.Dropout(dropout_rate, name="bottleneck_digits_do_1")(z1)

    # z1 = layers.Dense(512, activation="relu", name="bottleneck_digits_dense_2")(z1)
    # z1 = layers.Dropout(dropout_rate, name="bottleneck_digits_do_2")(z1)

    # card_type = layers.Dense(1, activation="sigmoid", name="card_type")(z)
    # cvv2_outputs = [layers.Dense(10, activation="softmax", name=f"cvv2_digit_{i}")(z1) for i in range(4)]
    # exp_date_outputs = [layers.Dense(10, activation="softmax", name=f"exp_date_digit_{i}")(z1) for i in range(8)]


    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    x1 = layers.SeparableConv2D(512, kernel_size=1, padding="same", use_bias=False, name="bboxes_anchors_bottleneck_conv_1")(x)
    x1 = layers.BatchNormalization(name="bboxes_anchors_bottleneck_bn_1")(x1)
    x1 = layers.ReLU(name="bboxes_anchors_bottleneck_relu_1")(x1)

    x1 = layers.SeparableConv2D(256, kernel_size=1, padding="same", use_bias=False, name="bboxes_anchors_bottleneck_conv_2")(x1)
    x1 = layers.BatchNormalization(name="bboxes_anchors_bottleneck_bn_2")(x1)
    x1 = layers.ReLU(name="bboxes_anchors_bottleneck_relu_2")(x1) 

    confs_outputs = [layers.SeparableConv2D(1, kernel_size=1, padding="same", activation="sigmoid", name=f"confs_anchor_{i}")(x) for i in range(num_anchors)]
    bboxes_outputs = [layers.SeparableConv2D(4, kernel_size=1, padding="same", activation="linear", name=f"bboxes_anchor_{i}")(x1) for i in range(num_anchors)]
    classes_outputs = [layers.SeparableConv2D(num_classes, kernel_size=1, padding="same", activation="softmax", name=f"classes_anchor_{i}")(x) for i in range(num_anchors)]


    model = models.Model(
        inputs=input_image, 
        outputs={
            **{f"confs_anchor_{i}": confs_outputs[i] for i in range(num_anchors)}, 
            **{f"bboxes_anchor_{i}": bboxes_outputs[i] for i in range(num_anchors)},
            **{f"classes_anchor_{i}": classes_outputs[i] for i in range(num_anchors)},
            # "card_type": card_type,
            # **{f"cvv2_digit_{i}": cvv2_outputs[i] for i in range(4)}, 
            # **{f"exp_date_digit_{i}": exp_date_outputs[i] for i in range(8)}, 
        }, 
        name=model_name
    )


    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, decay=reg_coef), 
        loss={
            **{f"confs_anchor_{i}": "binary_crossentropy" for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": "mse" for i in range(num_anchors)},
            **{f"classes_anchor_{i}": "sparse_categorical_crossentropy" for i in range(num_anchors)},
            # "card_type": "binary_crossentropy", 
            # **{f"cvv2_digit_{i}": "sparse_categorical_crossentropy" for i in range(4)}, 
            # **{f"exp_date_digit_{i}": "sparse_categorical_crossentropy" for i in range(8)}, 
        },
        metrics={
            **{f"confs_anchor_{i}": ["accuracy"] for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": ["mae"] for i in range(num_anchors)},
            **{f"classes_anchor_{i}": ["accuracy"] for i in range(num_anchors)},
            # "card_type": ["accuracy"], 
            # **{f"cvv2_digit_{i}": ["accuracy"] for i in range(4)}, 
            # **{f"exp_date_digit_{i}": ["accuracy"] for i in range(8)}, 
        },
        loss_weights={
            **{f"confs_anchor_{i}": 1. for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": 1. for i in range(num_anchors)},
            **{f"classes_anchor_{i}": 1. for i in range(num_anchors)},
            # "card_type": 0.1,
            # **{f"cvv2_digit_{i}": 0.5 for i in range(4)}, 
            # **{f"exp_date_digit_{i}": 0.5 for i in range(8)}, 
        },
    )


    return model


def load_model(model_path=model_path):
    return models.load_model(model_path, custom_objects={"OCRModel": OCRModel})


