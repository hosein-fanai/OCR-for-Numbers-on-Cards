from tensorflow.keras import callbacks

from mean_metric_callback import MeanMetricCallback

from constants import training_phase_2, model_path, monitor_metric, log_dir


def get_callbacks_list():
    callbacks_list = \
        ([MeanMetricCallback(metric_name="cvv2", metric_types=["loss", "accuracy"], post_name="digit", metric_num=4)] if training_phase_2 else []) + \
        ([MeanMetricCallback(metric_name="exp_date", metric_types=["loss", "accuracy"], post_name="digit", metric_num=8)] if training_phase_2 else []) + \
        [MeanMetricCallback(metric_name="confs", metric_types=["loss", "accuracy"]),
        MeanMetricCallback(metric_name="bboxes", metric_types=["loss", "mse"]),
        MeanMetricCallback(metric_name="classes", metric_types=["loss", "accuracy"]),
        callbacks.ModelCheckpoint(
            model_path, 
            monitor=monitor_metric, 
            save_best_only=True, 
            min_delta=0.0,
            verbose=0,
        ),
        callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=10,
            min_delta=1e-3,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=3,
            verbose=0,
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            write_graph=False,
        ),
    ]

    return callbacks_list


