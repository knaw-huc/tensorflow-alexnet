import numpy as np
import os
import math
from PIL import Image
from sys import getsizeof

def __get_resize_image__(image_path: str, image_size:int, grayscale:bool) -> np.ndarray:
    image = Image.open(image_path)

    if image.height > image.width:
        scale = image_size / image.height
    else:
        scale = image_size / image.width

    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    image = image.resize(size=(new_width, new_height))
    if grayscale:
        image = image.convert('L')

#    image = image.resize(size=(image_size, image_size))
    padded_image = Image.new("RGB", (image_size, image_size), "BLACK")
#todo: this needs to be centered
    padded_image.paste(image)
    padded_image.convert('RGB')
#    file = image_path.replace("/","_")
#    padded_image.save("/tmp/"+file+".jpg", "JPEG")
    padded_image = (np.array(padded_image, dtype=np.float16) / 255) - 0.5
#    print (getsizeof(padded_image))
    return padded_image

def reshape_example(example: (str, int), image_size:int, grayscale:bool) -> (np.ndarray, int):
    result = (__get_resize_image__(example[0], image_size, grayscale), example[1])
    return result

def get_images_from(dir: str, folder: str, file:str) -> [(str, int)]:
    data = []
    with open(os.path.join(dir, folder, file)) as labels:
        line = labels.readline()
        while line:
            (image, label_str) = line.split(" ")
            label = int(label_str)
            data.append((os.path.join(dir, "images", image), label))
            line = labels.readline()
    return data

def determine_number_of_classes(data:[(str, int)]) -> int:
    # highest label + 1 for zero indexed array
    return max(list(map(lambda item: item[1], data)))  + 1

class RvlCdip:
    def __init__(self, dir="/data/rvl-cdip/", batch_size=10, image_size = 224, grayscale=True):
        self.__res__ = get_images_from(dir, "labels", "train.txt")[:80000]
#        self.__res__ = get_images_from(dir, "labels", "train.txt")[:5000]
#        self.__res__ = get_images_from(dir, "labels", "train.txt")[:500]
        self.number_of_classes = determine_number_of_classes(self.__res__)
        self.batch_size = batch_size
        self.batches = []
        self.__batch_num__ = 0
        self.image_size = image_size
        self.grayscale = grayscale

        for i in range(math.ceil(len(self.__res__)/batch_size)):
            batch = self.__res__[i*batch_size:(i+1)*batch_size]
            new_batch = []
            print ("batch: "+ str(i))
            print ("images loaded: "+ str(i *batch_size))
            for j in range (len(batch)):
                item = list(batch[j])
                reshaped = reshape_example(item, self.image_size, self.grayscale)
#                print(reshaped)
                new_batch.append(reshaped)
            self.batches.append(new_batch)
#            self.batches.append(self.__res__[i*batch_size:(i+1)*batch_size])


        self.__test_set__ = get_images_from(dir, "labels", "val.txt")[:1000]
        new_batch = []
        for i in range(len(self.__test_set__)):
            item = list(self.__test_set__[i])
            reshaped = reshape_example(item, self.image_size, self.grayscale)
#            print(item)
            new_batch.append(reshaped)
        self.test_set = list(new_batch)
#        self.test_set = get_images_from(dir, "labels", "val.txt")

    def batch(self, num) -> [(np.ndarray, int)]:
        return list(self.batches[num])
#        return list(map(lambda example: reshape_example(example, self.image_size), self.batches[num]))


    def next_batch(self) -> [(np.ndarray, int)]:
        if self.__batch_num__ <= len(self.batches):
            res = self.batch(self.__batch_num__)
            self.__batch_num__ += 1
        else:
            res: []

        return res

    def reset_batch(self):
        self.__batch_num__ = 0

    def test_set(self):
#        return list(self.batches[0])
        return self.test_set
#        return list(map(lambda example: reshape_example(example, self.image_size), self.__test_set__))
