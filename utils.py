import os
import cv2
import numpy as np
import pytesseract

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
custom_config = r"--oem 3 --psm 6"


def input_file(filename):
    return os.path.join(INPUT_FOLDER, filename)


def output_file(filename):
    return os.path.join(OUTPUT_FOLDER, filename)


def mostrar(img, PPimg):
    cv2.imshow("PPimg", redimensionar(PPimg, 1 / 8))
    cv2.imshow("Original", redimensionar(img, 1 / 2))


def log(arquivo, PPimg):
    f = open(output_file(arquivo + ".txt"), "w")
    f.write(pytesseract.image_to_string(PPimg, config=custom_config))
    f.close()


def blur(img, metodo):
    match metodo:
        case 1:
            return cv2.blur(img, ksize=(9, 9))
        case 2:
            return cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=30, sigmaY=30)
        case 3:
            return cv2.medianBlur(img, 5)


def nitidizar(img):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    return cv2.filter2D(img, -1, k)


def redimensionar(img, size):
    return cv2.resize(img, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)


def arestas(img, t1, t2):
    return cv2.Canny(img, t1, t2)


def dilatar(img):
    k = np.ones((5, 5), np.uint8)
    return cv2.dilate(img, k, iterations=1)


def erodir(img):
    k = np.ones((5, 5), np.uint8)
    return cv2.erode(img, k, iterations=1)


def limiarizar(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def preProcessamento(img):
    img = redimensionar(img, 4)
    # img = nitidizar(img)
    # img = blur(img, 2)
    img = limiarizar(img)
    # img = dilatar(img)
    # img = erodir(img)
    # img = arestas(img, 105, 127)
    return img
