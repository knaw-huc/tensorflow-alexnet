import numpy as np
import os
import math
from PIL import Image

def __get_resize_image__(image_path: str, image_size:int) -> np.ndarray:
    image = Image.open(image_path)

    if image.height > image.width:
        scale = image_size / image.height
    else:
        scale = image_size / image.width

    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    image = image.resize(size=(new_width, new_height))

    padded_image = Image.new("RGB", (image_size, image_size), "BLACK")
    padded_image.paste(image)
    padded_image.convert('RGB')

    return np.array(padded_image, ndmin=3) / 255

def reshape_example(example: (str, int), image_size:int) -> (np.ndarray, int):
    result = (__get_resize_image__(example[0], image_size), example[1])
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
    def __init__(self, dir="/data/rvl-cdip/", batch_size=10):
        self.__res__ = get_images_from(dir, "labels", "train.txt")
        self.number_of_classes = determine_number_of_classes(self.__res__)
        self.batch_size = batch_size
        self.batches = []
        self.__batch_num__ = 0
        self.image_size = 32

        for i in range(math.ceil(len(self.__res__)/batch_size)):
            self.batches.append(self.__res__[i*batch_size:(i+1)*batch_size])

        self.__test_set__ = get_images_from(dir, "labels", "val.txt")

    def batch(self, num) -> [(np.ndarray, int)]:
        return list(map(lambda example: reshape_example(example, self.image_size), self.batches[num]))


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
        return list(map(lambda example: reshape_example(example, self.image_size), self.__test_set__))
