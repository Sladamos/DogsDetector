import os

from models.comparators.CNNModelComparator import CNNModelComparator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

import matplotlib.pyplot as plt


from models.TensorCNNModel import TensorCNNModel
from models.creators.TensorLayersCreator import TensorLayersCreator


cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)  # 50000, 32, 32, 3

# Normalize: 0,255 -> 0,1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def show():
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


#show()

# model...


# import sys; sys.exit()

# loss and optimizer

# training
batch_size = 32
epochs = 3
model.train(train_images, train_labels, epochs=epochs,
          number_of_samples=batch_size, verbose=1)

# evaulate
result = model.evaluate(test_images, test_labels, number_of_samples=batch_size)
print(result)
comparator = CNNModelComparator(test_images, test_labels)




# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)

x_train = train_images
x_test = test_images
y_train = train_labels
y_test = test_labels

# model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

model.compile(loss=loss, optimizer=optim, metrics=metrics)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

print(model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1))