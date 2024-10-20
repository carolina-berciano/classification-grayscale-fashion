import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import math
import numpy as numpy
import matplotlib.pyplot as plt

tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# import data
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# get training data inputs and test data inputs
train_dataset, test_dataset = dataset['train'], dataset['test']

print('metadata: ', metadata)

# get labels
class_names = metadata.features['label'].names
print(class_names)

# explore data
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print('Number of train examples: ', num_train_examples, 'Number of test examples: ', num_test_examples)


# preprocess data
# normalize value of pixels
def normalize(images, labels):
    images = tf.cast(images, tf.float32)  # convert from int to tf float
    images /= 255
    return images, labels


# map function applies a given function to each item in the dataset
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# keep the dataset in memory after first time loaded (speed up training)
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# Plot images and verify they are valid (-- optional--)
plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(train_dataset.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()

# building the model
# set up the layers
model = tf.keras.Sequential([
    # creates 32 convoluted images from 32 different filters applied to the input image, padding to retain the original image size
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    # reduce in size the 32 convoluted images
    tf.keras.layers.MaxPool2D((2, 2), strides=2),
    # takes the 32 images as input and creates 64 outputs
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    # reduce in size the 64 convoluted images
    tf.keras.layers.MaxPool2D((2, 2), strides=2),
    # transform image from 28x28px 2d-array to 1d-array of 784px (no neurons to learn, only reformat the data)
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # 128 neurons, each connected to each of the 784 nodes from previous layer, relu
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # 10 neurons, each neuron for each potential class, the softmax activation function create the probabilistic distribution
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# train the model
# prepare dataset for model. repeat defines infinite iteration until epochs is set, shuffle to prevent model from
# learning specific positions and batch to train in batch of images
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# evaluate model
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset: ', test_accuracy)

# make predictions (in batch)
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

print('predictions shape', predictions.shape)

print('Predictions for first test example: ', predictions[0])

print('Highest confidence value: ', np.argmax(predictions[0]))

print('Verify real test label: ', test_labels[0])

# make predictions (single image)
img = test_images[0]
print("Single image shape :", img.shape)

# Convert it to array so that can be pass to predict method
img = np.array([img])
print("Single image arr shape :", img.shape)

predictions_single = model.predict(img)

print('single predictions: ', predictions_single)

print('Highest confidence value: ', np.argmax(predictions_single[0]))


