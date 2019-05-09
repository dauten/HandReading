from __future__ import absolute_import, division, print_function
from PIL import Image
import time
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
         10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z',
         36:'A',37:'B',38:'C',39:'D',40:'E',41:'F',42:'G',43:'H',44:'I',45:'J',46:'K',47:'L',48:'M',49:'N',50:'O',51:'P',52:'Q',53:'R',54:'S',55:'T',56:'U',57:'V',58:'W',59:'X',60:'Y',61:'Z'}

lowercase_dict = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'}

#iterates each side of image to find edges of letter
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

#gets key for result
def getResult(res):
    i = 0
    for r in res:
        if(r == 1):
            return i
        i+=1

class interpreter:

    #Loads the model into the class. Change filepath for different models
    def __init__(self):
        self.model =  keras.models.load_model('HandWritingNN2.h5')

    #Takes in an image and preprocesses it before predicting what it is
    def eval(self, img):
        img = img.resize((128,128))

        #finding edge of letter for cropping.
        img2 = img.load()
        dim = findDem(img2)

        #crops and resizes image
        img = img.crop((dim[0],dim[1],dim[2],dim[3]))
        img = img.resize((28,28))

        #create numpy array with image for model
        imgx = np.array(img.getdata()).reshape(img.size[0], img.size[1])
        imgarray = np.ndarray((1,28,28))
        imgarray[0] = imgx

        result = self.model.predict(imgarray)

        return lowercase_dict[getResult(result[0])]
