import os

import cv2


def getListofFiles(dir):
    inputs = []

    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            inputs.append(path)
    return inputs

def getImages(dir):
    images = []
    inp = getListofFiles(dir)
    for i in range(len(inp)):
        temp = cv2.imread(dir + inp[i])
        images.append(temp)
    return images