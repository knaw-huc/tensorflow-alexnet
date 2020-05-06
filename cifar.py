import pickle
import numpy as np
import os
import math

def __extract_file__(fname:str):
    with open(fname, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def __deflate_image__(flat_image):
    red = flat_image[0:1024].reshape((32,32))
    green = flat_image[1024:2048].reshape((32,32))
    blue = flat_image[2048:3072].reshape((32,32))

    image = np.dstack((red, green, blue))

    return image/255

def __extract_reshape_file__(fname:str):
    result = []
    data = __extract_file__(fname)
    images = data[b"data"]
    labels = data[b"labels"]

    for image, label in zip(images, labels):
        result.append((__deflate_image__(image), label))
    
    return result

def get_images_from(dir):
    files = [f for f in os.listdir(dir) if f.startswith("data_batch")]
    res = []

    for f in files:
        res = res + __extract_reshape_file__(os.path.join(dir, f))
    
    return res

class Cifar:
    def __init__(self, dir="/data/cifar-10-batches-py/", batch_size=1):
        self.__res__ = get_images_from(dir)
        self.batch_size = batch_size
        self.batches = []
        self.__batch_num__ = 0

        for i in range(math.ceil(len(self.__res__)/batch_size)):
            self.batches.append(self.__res__[i*batch_size:(i+1)*batch_size])
        
        self.test_set = __extract_reshape_file__(os.path.join(dir, "test_batch")) # <- Added for test data
    
    def batch(self, num):
        return self.batches[num]
    
    def next_batch(self):
        if self.__batch_num__ <= len(self.batches):
            res = self.batches[self.__batch_num__]
            self.__batch_num__ += 1
        else:
            res: []

        return res
    
    def reset_batch(self):
        self.__batch_num__ = 0
