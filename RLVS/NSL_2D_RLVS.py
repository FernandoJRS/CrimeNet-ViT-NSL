import neural_structured_learning as nsl
import time
import datetime
from Data_RLVS import *


vit_model = create_vit_classifier()
vit_model.summary()

adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2,
                                             adv_step_size=0.05,
                                             adv_grad_norm='infinity')

adv_model = nsl.keras.AdversarialRegularization(vit_model,
                                                label_keys=['label'],
                                                adv_config=adv_config)

adv_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[# tf.keras.metrics.TruePositives(name='tp'),
                           # tf.keras.metrics.FalsePositives(name='fp'),
                           # tf.keras.metrics.TrueNegatives(name='tn'),
                           # tf.keras.metrics.FalseNegatives(name='fn'),
                           # tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                           tf.keras.metrics.AUC(curve="ROC"),
                           tf.keras.metrics.AUC(curve="PR")])

log_dir = "/home/user/work/shared-CrimeNet/RLVS/Results/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "/home/user/work/shared-CrimeNet/RLVS/Results/logs/checkpoint/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

start_time_train = time.time()
adv_model.fit(generatorTrainData(batch_size_train=32),
              epochs=10,
              steps_per_epoch=int(len(train_total_op) / 32),
              validation_data=generatorValidationData(batch_size_train=32),
              validation_steps=int(len(validation_total_op) / 32),
              callbacks=[tensorboard_callback, cp_callback],
              shuffle=True)
print('Training time per epoch: ' + str((time.time() - start_time_train) / 10))

start_time_test = time.time()
adv_model.evaluate(generatorTestData(batch_size_test=32),
                   steps=int(len(test_total_op) / 32))
print('Inference time: ' + str((time.time() - start_time_test) / len(test_total_op)))