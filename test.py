#quick test program to try out tensorflow
#taken from https://www.tensorflow.org/tutorials/keras/basic_classification
#trying to just get a basic view of image classification
#since this is essentially what our project is



from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


#these lines download the dataset

#we can use the fashion dataset or the handwritten digit (0-9) dataset
#fashion_mnist = keras.datasets.fashion_mnist
fashion_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#this prints the first test data (datum?) and its label
#this is an array map of grayscale values and the corresponding label (9, so ankle boot)
for x in train_images[0]:
    for y in x:
        print(str('%03d' % y)+" ", end="")
    print("\n")

print(train_images[0][12])
print(train_labels[0])

#this is just a map so printing is easier
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(class_names[train_labels[0]])

'''
#this prints the first training data for visualization purposes
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

#this is some preprocessing, it convert each pixel value to a decimal where 0.0 = no value and 1 = max value
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
#Let's view it again to see if it changed.  The image didn't but the values and scale did
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

#this does more processing to the image, cconverting it from a 28x28 matrix to a 784 length array.
#the dense commands actually create neurons/nodes
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

#this tells the neural net how to update things for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#this is the actual 'train' step
#you can play with epochs, increasing it takes longer but provides slightly more accuracy
model.fit(train_images, train_labels, epochs=5)


#this runs our test dataset through our network and tells us how accurate it was
#I got about 87.5% (varies from run to run) so that's not bad for an elementary first try
#with epochs=100 I got accuracy of 88%, it seems like increasing breadth of
#training data will help us more than increasing work the neural net does cause that was a
#garbage increase for taking 20x as long
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

'''
Where to go from here?
First we'll need a simple handwritting dataset (https://www.nist.gov/node/1298471/emnist-dataset may work?).  We'll use that with labels to train a new
neural net and then we'll have something that can recognize individual characters.
After that we'll just need to find a way to look at an image with handwritting and convert it
to a series of character images for us to process (openCV for this might work well)
'''
