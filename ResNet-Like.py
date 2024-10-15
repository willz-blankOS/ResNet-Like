import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

ds_train, ds_test = keras.datasets.cifar10.load_data()

(x_train, y_train) = ds_train
(x_test, y_test) = ds_test

x_train = x_train.reshape(-1, 32, 32, 3).astype("float32") / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype("float32") / 255.0

class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, padding='same', regularizer=0):
        super(CNNBlock, self).__init__()
        self.cnn = layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, kernel_regularizer=keras.regularizers.l2(regularizer))
        self.bn = layers.BatchNormalization()
    
    def call(self, input_tensor, training=False):
        input = self.cnn(input_tensor, training=training)
        x = self.bn(input, training=training)
        x = tf.nn.relu(x)
        return x

class ResBlock(layers.Layer):
    def __init__(self, filters=[32,64,128], regularizers=0):
        super(ResBlock, self).__init__()
        self.cnn1 = CNNBlock(filters=filters[0], regularizer=regularizers)
        self.cnn2 = CNNBlock(filters=filters[1], regularizer= regularizers)
        self.cnn3 = CNNBlock(filters=filters[2], regularizer=regularizers)
        self.identity_mapping = layers.Conv2D(filters=filters[1], kernel_size=1, strides=1, padding='same')
        self.pooling = layers.MaxPooling2D()

    def call(self, input_tensor, training=False):
        input = self.cnn1(input_tensor, training=training)
        x = self.cnn2(input, training=training)
        x = self.cnn3(
            x + self.identity_mapping(input_tensor), training=training
        )
        return self.pooling(x)
    
class ResNet_Model(keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_Model, self).__init__()
        self.block1 = ResBlock(filters=[32,32,64], regularizers=0.02)
        self.block2 = ResBlock(filters=[64,128,256], regularizers=0.02)
        self.block3 = ResBlock(filters=[128, 256, 256], regularizers=0.02)
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        input = self.block1(input_tensor, training=training)
        x = self.block2(input, training)
        x = self.block3(x, training=training)
        x = self.pool(x)
        return self.classifier(x)

    def model(self):
        x = keras.Input(shape=(32,32,3))
        return keras.Model(inputs=[x], outputs=self.call(x))
    

model = ResNet_Model(num_classes=10)
print(model.model().summary())

num_epochs = 10
lr = 1e-3
optimizer = keras.optimizers.Adam()
losses = keras.losses.SparseCategoricalCrossentropy(lr)
metric = keras.metrics.SparseCategoricalAccuracy()

print(f"\n-------------- TRAINING MODEL --------------")

#Training loop
for epoch in range(num_epochs):
    print(f"\nTraining Epoch {epoch}/{num_epochs}")
    for batch_idx, (x_batch, y_batch) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = losses(y_batch, y_pred)
            
        gradient = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradient, model.trainable_weights))
        metric.update_state(y_batch, y_pred)
        
    train_acc = metric.result()
    print(f"\nAccuracy: {train_acc}")
    metric.reset_states()

print(f"\n\n-------------- TRAINING COMPLETE --------------")
print(f"\nEvaluating data...\n")

#Test loop
for batch_idx, (x_batch, y_batch) in enumerate(ds_test):
    y_pred = model(x_batch, training=True)
    metric.update_state(y_batch, y_pred)
    print(f"\nEvaluating Batch {batch_idx}...")
    
print(f"Evaluation complete...")
train_acc = metric.result()
print(f"\nModel Accuracy: {train_acc}")