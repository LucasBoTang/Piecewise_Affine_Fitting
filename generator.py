#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2

def generate_images(size):
    """
    generate depth image
    """
    images = []

    # image 0
    image = np.zeros((4*size, 6*size)) + 0.3
    grid = np.zeros((4*size, 6*size))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i // size) % 2 and (j // size) % 2:
                image[i, j] = 0.7
                grid[i, j] = 1
            if not (i // size) % 2 and not (j // size) % 2:
                image[i, j] = 0.7
                grid[i, j] = 1
    for j in range(image.shape[1]):
        image[:, j] -= 0.03 * grid[:, j] * j / size
        image[:, j] += 0.02 * (1 - grid[:, j]) * j / size
    images.append((0, image))

    # image 1
    image = np.zeros((4*size, 6*size)) + 0.1
    image[0,0] = 0.0
    for j in range(1, 4*size):
        image[0,j] = image[0,j-1] + (0.15 / size)
    for i in range(1, 1*size):
        image[i,:4*size] = image[i-1,:4*size] + (0.1 / size)
    for i in range(1*size, 2*size):
        image[i,:3*size] = image[i-1,:3*size] + (0.1 / size)
    image[0, 4*size] = 1
    for i in range(3*size):
        for j in range(4*size, 5*size):
            image[i, j] = 1 - i * (0.05 / size) - (j - size * 4)* (0.4 / size)
    for i in range(size//2):
        for j in range(1*size-i):
            image[3*size+i+j, 1*size+j] = 0.7
    images.append((1, image))

    # image 2
    image = np.zeros((4*size, 6*size)) + 0.5
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i + j > 47 * size // 10:
                continue
            image[i, j] = 0.4 + (i + j) % (16 * size // 10) * (0.3 / size)
    image[2*size, 4*size] = 0.4
    for i in range(2*size+1, 3*size+size//2):
        image[i, 4*size] = image[i-1, 4*size] - (0.08 / size)
    for j in range(4*size+1, 5*size+size//2):
        image[2*size:3*size+size//2, j] = image[2*size:3*size+size//2, j-1] - (0.06 / size)
    images.append((2, image))

    # image 3
    disp = cv2.imread("./data/image3.png", 0)
    image = disp_to_depth(disp)
    # get size
    h, w = image.shape
    ratio = np.sqrt(24 * size ** 2 / (h * w))
    h, w = int(h * ratio), int(w * ratio)
    image = cv2.resize(image, (w, h))
    images.append((3, image))

    disp = cv2.imread("./data/image4.png", 0)
    image = disp_to_depth(disp)
    # get size
    h, w = image.shape
    ratio = np.sqrt(24 * size ** 2 / (h * w))
    h, w = int(h * ratio), int(w * ratio)
    image = cv2.resize(image, (w, h))
    images.append((4, image))

    disp = cv2.imread("./data/image5.png", 0)
    image = disp_to_depth(disp)
    # get size
    h, w = image.shape
    ratio = np.sqrt(24 * size ** 2 / (h * w))
    h, w = int(h * ratio), int(w * ratio)
    image = cv2.resize(image, (w, h))
    images.append((5,image))

    return images

def disp_to_depth(disp):
    """
    convert disparity map into depth map
    """
    # remove invalid pixel
    mask = (disp > 64)
    # reverse
    depth = 1 / disp * mask
    depth += (1 - mask) * np.nanmax(depth)
    # rescale
    depth /= np.nanmax(depth)
    # fill na
    depth[np.isnan(depth)] = 1

    return depth

if __name__ == "__main__":
    image = generate_images(20)[2]
