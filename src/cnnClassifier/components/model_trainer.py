import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
# import tensorflow_addons as tfa
# import tensorflow_model_optimization as tfmot


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )


    # def prune_model(self):
    #     self.model.trainable=True
    #     self.prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude(self.model)


    #     batch_size = 64
    #     validation_split = 0.2
    #     epochs=30

    #     # num_images = train_df_merged_copy.shape[0] * (1 - validation_split)

    #     # # end_step = np.ceil(num_images / batch_size).astype(np.int8) * epochs
    #     # end_step = int(np.ceil(num_images / batch_size)) * epochs
    #     # print(end_step)


    #     # Define model for pruning.
    #     '''
    #     start the model with 50% sparsity (50% zeros in weights) and end with 80% sparsity.
    #     '''
    #     pruning_params = {
    #         'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
    #                                                                 final_sparsity=0.80,
    #                                                                 begin_step=0,
    #                                                                 end_step=200,
    #                                                                 frequency=100
    #                                                                 )
    #     }

    #     self.model_for_pruning = self.prune_low_magnitude(self.model, **pruning_params)

    #     '''
    #     tfmot.sparsity.keras.prune_low_magnitude ---> will apply pruning to the whole model
    #     '''

    #     f1 = tfa.metrics.F1Score(num_classes=num_classes, average='macro')
    #     reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=10)
    #     checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/ec2-user/SageMaker/QAT/MobileNet/MobileNet_035/Fine_tuning/pruned_model.h5', 
    #                                                     monitor='val_accuracy', 
    #                                                     mode='max', 
    #                                                     verbose=1, 
    #                                                     save_best_only=True)
    #     logdir = tempfile.mkdtemp()

    #     '''
    #     UpdatePruningStep-
    #     Keras callback which updates pruning wrappers with the optimizer step.

    #     PruningSummaries-
    #     provides logs for tracking progress and debugging
    #     '''

    #     updatePruningstep = tfmot.sparsity.keras.UpdatePruningStep(),
    #     pruningLog = tfmot.sparsity.keras.PruningSummaries(log_dir=logdir, profile_batch=0),

    #     tqdm_callback = tfa.callbacks.TQDMProgressBar()
    #     optimizer = Adam(learning_rate=0.00001)

    #     model_for_pruning.compile(optimizer=optimizer, 
    #                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
    #                             metrics=[['accuracy'],f1])
    #     print(model_for_pruning.summary())

    #     pruning_history = model_for_pruning.fit(train_df, epochs=50, 
    #                                 validation_data=val_df,
    #                                 verbose=1,
    #                                 callbacks=[reduce_lr, checkpoint, updatePruningstep, pruningLog])