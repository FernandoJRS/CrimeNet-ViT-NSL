import neural_structured_learning as nsl
import time
import datetime
from Data_UBI import *

vit_model = create_vit_classifier()
vit_model.summary()

adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2,
                                             adv_step_size=0.05,
                                             adv_grad_norm='infinity')

adv_model = nsl.keras.AdversarialRegularization(vit_model,
                                                label_keys=['label'],
                                                adv_config=adv_config)

adv_model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[#keras.metrics.TruePositives(name='tp'),
                           #keras.metrics.FalsePositives(name='fp'),
                           #keras.metrics.TrueNegatives(name='tn'),
                           #keras.metrics.FalseNegatives(name='fn'),
                           #keras.metrics.CategoricalAccuracy(name='accuracy'),
                           keras.metrics.AUC(curve="ROC"),
                           keras.metrics.AUC(curve="PR")])

log_dir = "Results/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "Results/logs/checkpoint/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

start_time_train = time.time()
adv_model.fit(generatorData(train_total_op, train_total_rgb, num_classes, batch_size=32),
              epochs=10,
              steps_per_epoch=int(len(train_total_op) / 32),
              validation_data=generatorData(test_total_op, test_total_rgb, num_classes, batch_size=32),
              validation_steps=int(len(test_total_op) / 32),
              callbacks=[tensorboard_callback, cp_callback],
              shuffle=True)
print('Training time per epoch: ' + str((time.time() - start_time_train) / 10))

start_time_test = time.time()
adv_model.evaluate(generatorData(test_total_op, test_total_rgb, num_classes, batch_size=32),
                   steps=int(len(test_total_op) / 32))
print('Inference time: ' + str((time.time() - start_time_test) / len(test_total_op)))

if __name__ == '__main__':
    pass
