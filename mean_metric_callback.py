from tensorflow.keras import callbacks

import numpy as np

from constants import num_anchors


class MeanMetricCallback(callbacks.Callback):
    
    def __init__(self, metric_name, metric_types):
        super(MeanMetricCallback, self).__init__()
        self.metric_name = metric_name
        self.metric_types = metric_types

        self.metrics_list = [[f"{self.metric_name}_anchor_{i}_{type_}" for i in range(num_anchors)]\
                                                                         for type_ in metric_types]

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        for metrics, metric_type in zip(self.metrics_list, self.metric_types):
            logs[self.metric_name+'_'+metric_type] = np.mean([logs[metric] for metric in metrics])
            logs["val_"+self.metric_name+'_'+metric_type] = np.mean([logs["val_"+metric] for metric in metrics])


