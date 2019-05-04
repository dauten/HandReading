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

training_directory = "../python_test/lowercase/"

training_images = list()
training_labels = list()

class_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
         10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z',
         36:'A',37:'B',38:'C',39:'D',40:'E',41:'F',42:'G',43:'H',44:'I',45:'J',46:'K',47:'L',48:'M',49:'N',50:'O',51:'P',52:'Q',53:'R',54:'S',55:'T',56:'U',57:'V',58:'W',59:'X',60:'Y',61:'Z'}

lowercase_dict = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'}

def get_key(val):
    for k,v in lowercase_dict.items():
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

#this is some preprocessing, it convert each pixel value to a decimal where 0.0 = no value and 1 = max value
train_images = train_images / 255.0

#this does more processing to the image, cconverting it from a 28x28 matrix to a 784 length array.
#the dense commands actually create neurons/nodes
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(26, activation=tf.nn.softmax)])

#this tells the neural net how to update things for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#this is the actual 'train' step
#you can play with epochs, increasing it takes longer but provides slightly more accuracy
model.fit(train_images, train_labels, epochs=50)


test_loss, test_acc = model.evaluate(train_images, train_labels)
print('Test accuracy:', test_acc)

tester = np.ndarray((1,28,28))
tester[0] = training_images[110000]
result = model.predict(tester)
print(str(lowercase_dict[getResult(result[0])]))

model.save("HandWritingNN2.h5")