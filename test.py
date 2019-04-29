#quick test program to try out tensorflow
#taken from https://www.tensorflow.org/tutorials/keras/basic_classification
#trying to just get a basic view of image classification
#since this is essentially what our project is



from __future__ import absolute_import, division, print_function

from PIL import Image
import time
import os

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
# fashion_mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# print(train_labels.shape)

training_directory = "../python_test/lowercase/"

training_images = list()
training_labels = list()

class_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
         10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z',
         36:'A',37:'B',38:'C',39:'D',40:'E',41:'F',42:'G',43:'H',44:'I',45:'J',46:'K',47:'L',48:'M',49:'N',50:'O',51:'P',52:'Q',53:'R',54:'S',55:'T',56:'U',57:'V',58:'W',59:'X',60:'Y',61:'Z'}

def get_key(val):
    for k,v in class_dict.items():
        if(v == val):
            return k

def findDem(t):
    u = 0
    r = 127
    d = 127
    l = 0

    while(u < 128):
        iter = 0
        b = False
        while(iter < 128):
            if(t[iter,u] == 0):
                b = True
                break
            else:
                iter+=1
        if(b==True):
            break
        else:
            u+=1

    while(r >= 0):
        iter = 0
        b = False
        while(iter < 128):
            if(t[r,iter] == 0):
                b=True
                break
            else:
                iter+=1
        if(b==True):
            break
        else:
            r-=1

    while(d >= 0):
        iter = 0
        b = False
        while(iter < 128):
            if(t[iter,d] == 0):
                b = True
                break
            else:
                iter+=1
        if(b==True):
            break
        else:
            d-=1

    while(l < 128):
        iter = 0
        b = False
        while(iter < 128):
            if(t[l,iter] == 0):
                b=True
                break
            else:
                iter+=1
        if(b==True):
            break
        else:
            l+=1

    return [l,u,r,d]

def getResult(res):
    i = 0
    for r in res:
        if(r == 1):
            return i
        i+=1


start = time.time()
for subdir, dirs, files in os.walk(training_directory):
    for file in files:
        temp = Image.open(subdir+'/'+file).convert("L")
        temp2 = temp.load()
        dim = findDem(temp2)
        temp = temp.crop((dim[0],dim[1],dim[2],dim[3]))
        temp = temp.resize((28,28))
        tempx = np.array(temp.getdata()).reshape(temp.size[0], temp.size[1])
        #tempx = tempx / 255.0
        training_images.append(tempx)
        training_labels.append(get_key(subdir[len(subdir) - 1]))
    print("Folder " + subdir[len(subdir) - 1] + " Done")
print("training images done in " + str(time.time() - start) + " seconds")

train_images = np.ndarray((len(training_images), 28, 28))
train_labels = np.ndarray((len(training_labels), 1))
k = 0
for i in training_images:
    train_images[k] = i
    k+=1

k = 0
for l in training_labels:
    train_labels[k] = l
    k+=1


#this prints the first test data (datum?) and its label
#this is an array map of grayscale values and the corresponding label (9, so ankle boot)
# for x in train_images[0]:
#     for y in x:
#         print(str('%03d' % y)+" ", end="")
#     print("\n")

# #this is just a map so printing is easier
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(class_names[train_labels[0]])

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
#test_images = test_images / 255.0


#Let's view it again to see if it changed.  The image didn't but the values and scale did
print(class_dict[train_labels[110000][0]])
plt.figure()
plt.imshow(train_images[110000])
plt.colorbar()
plt.grid(False)
plt.show()

#this does more processing to the image, cconverting it from a 28x28 matrix to a 784 length array.
#the dense commands actually create neurons/nodes
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(62, activation=tf.nn.softmax)])

#this tells the neural net how to update things for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#this is the actual 'train' step
#you can play with epochs, increasing it takes longer but provides slightly more accuracy
model.fit(train_images, train_labels, epochs=65)


#this runs our test dataset through our network and tells us how accurate it was
#I got about 87.5% (varies from run to run) so that's not bad for an elementary first try
#with epochs=100 I got accuracy of 88%, it seems like increasing breadth of
#training data will help us more than increasing work the neural net does cause that was a
#garbage increase for taking 20x as long
test_loss, test_acc = model.evaluate(train_images, train_labels)
print('Test accuracy:', test_acc)

tester = np.ndarray((1,28,28))
tester[0] = training_images[110000]
result = model.predict(tester)
print(str(class_dict[getResult(result[0])]))

model.save("HandWritingNN.h5")

'''
Where to go from here?
First we'll need a simple handwritting dataset (https://www.nist.gov/node/1298471/emnist-dataset may work?).  We'll use that with labels to train a new
neural net and then we'll have something that can recognize individual characters.
After that we'll just need to find a way to look at an image with handwritting and convert it
to a series of character images for us to process (openCV for this might work well)
'''
