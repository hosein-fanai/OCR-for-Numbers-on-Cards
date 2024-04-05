import tensorflow as tf
from tensorflow.keras import models, losses

from ocr.init import num_classes # , class_weights_obj


class OCRModel(models.Model):

    def train_step(self, data):
        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            if not self.compiled_loss.built:
                self.compiled_loss.build(predictions)
            
            total_loss = tf.reduce_sum(self.losses)
            for output_name, loss_fn_name in self.loss.items():
                loss_fn = losses.get(loss_fn_name)

                if "bboxes_anchor_" in output_name:
                    anchor_index = int(output_name[-1])
                    masks = tf.tile(targets[f"confs_anchor_{anchor_index}"], [1, 1, 1, 4])
                    loss = loss_fn(targets[output_name], predictions[output_name] * masks)
                    # loss *= masks * class_weights_obj # needs to be done before the aggregation of loss function
                elif "classes_anchor_" in output_name:
                    anchor_index = int(output_name[-1])
                    masks = tf.tile(targets[f"confs_anchor_{anchor_index}"], [1, 1, 1, num_classes])
                    loss = loss_fn(targets[output_name], predictions[output_name] * masks)
                    # loss *= masks * class_weights_obj
                elif "confs_anchor_" in output_name:
                    loss = loss_fn(targets[output_name], predictions[output_name])
                    # loss *= masks * class_weights_obj
                else:
                    loss = loss_fn(targets[output_name], predictions[output_name])

                loss *= self.compiled_loss._user_loss_weights[output_name]
                loss = tf.reduce_mean(loss)
                self.compiled_loss.metrics[list(self.loss.keys()).index(output_name)+1].update_state(loss)
                total_loss += loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_loss.metrics[0].update_state(total_loss)
        self.compiled_metrics.update_state(targets, predictions)
        metrics_dict = {m.name: m.result() for m in self.metrics}

        return metrics_dict


