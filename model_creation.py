import tensorflow as tf 
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras import backend as K

from ocr_model import OCRModel
from constants import input_shape, dropout_rate, num_anchors, num_classes, model_name, lr, reg_coef


def create_sliding_window_ocr_model():
    K.clear_session()


    conv_base = applications.Xception(include_top=False, input_shape=input_shape)
    for layer in conv_base.layers[:-15]:
        layer.trainable = False

    input_image = layers.Input(shape=input_shape, dtype=tf.uint8, name="image")
    
    x = models.Sequential([
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation_layer")(input_image)

    x = layers.Lambda(lambda X: applications.xception.preprocess_input(X), name="xception_preprocess")(x)

    x = conv_base(x)


    z = layers.GlobalAveragePooling2D()(x)

    z = layers.Dense(1024, activation="relu", name="bottleneck_dense_1")(z)
    z = layers.Dropout(dropout_rate, name="bottleneck_do_1")(z)

    z = layers.Dense(512, activation="relu", name="bottleneck_dense_2")(z)
    z = layers.Dropout(dropout_rate, name="bottleneck_do_2")(z)

    card_type = layers.Dense(1, activation="sigmoid", name="card_type")(z)
    # cvv2 = layers.Dense(1, activation="linear", name="cvv2")(z)
    # exp_date = layers.Dense(1, activation="linear", name="exp_date")(z)


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
            "card_type": card_type,
            # "cvv2": cvv2,
            # "exp_date": exp_date
        }, 
        name=model_name
    )


    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, decay=reg_coef), 
        loss={
            **{f"confs_anchor_{i}": "binary_crossentropy" for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": "mean_squared_error" for i in range(num_anchors)},
            **{f"classes_anchor_{i}": "sparse_categorical_crossentropy" for i in range(num_anchors)},
            "card_type": "binary_crossentropy", 
            # "cvv2": "mean_squared_error", 
            # "exp_date": "mean_squared_error",
        },
        metrics={
            **{f"confs_anchor_{i}": ["accuracy"] for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": ["mae"] for i in range(num_anchors)},
            **{f"classes_anchor_{i}": ["accuracy"] for i in range(num_anchors)},
            "card_type": ["accuracy"], 
            # "cvv2": ["mae"], 
            # "exp_date": ["mae"],
        },
        loss_weights={
            **{f"confs_anchor_{i}": 1. for i in range(num_anchors)},
            **{f"bboxes_anchor_{i}": 1. for i in range(num_anchors)},
            **{f"classes_anchor_{i}": 1. for i in range(num_anchors)},
            "card_type": 0.5,
            # "cvv2": 1.,
            # "exp_date": 1.,
        },
    )


    return model


