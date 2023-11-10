import os
import sys
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


def mostrar(img, PPimg, arquivo):
    match arquivo:
        case "IMG_0122":
            # cv2.imshow("PPimg", redimensionar(PPimg, (1 / 8)))
            cv2.imshow("PPimg", PPimg)
            # cv2.imshow("Original", redimensionar(img, (1 / 2)))
        case "MobPhoto_1":
            pass
        case "MobPhoto_5":
            cv2.imshow("PPimg", redimensionar(PPimg, (1 / 9)))
            cv2.imshow("Original", redimensionar(img, (1 / 3)))


def log(PPimg, arquivo):
    extract = pytesseract.image_to_string(PPimg, config=custom_config)
    f = open(output_file(arquivo + ".txt"), "w")
    f.write(extract)
    f.close()
    print(extract)


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


def dilatar(img, i):
    k = np.ones((3, 85), np.uint8)
    return cv2.dilate(img, k, iterations=i)


def erodir(img, i):
    k = np.ones((5, 5), np.uint8)
    return cv2.erode(img, k, iterations=i)


def limiarizar(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def corrigirPerspectiva(img, arquivo):
    match arquivo:
        case "MobPhoto_1":
            pass
        case "MobPhoto_5":
            pts1 = np.float32([[280, 180], [1625, 210], [47, 1520], [1860, 1520]])
            pts2 = np.float32([[0, 0], [1536, 0], [0, 2048], [1536, 2048]])
            matriz = cv2.getPerspectiveTransform(pts1, pts2)

            return cv2.warpPerspective(img, matriz, (1536, 2048))


def preProcessamento(img, arquivo):
    match arquivo:
        case "IMG_0122":
            h, w, c = img.shape

            if w > 1000:
                novo_w = 1000
                ar = w / h
                novo_h = int(novo_w / ar)
                img = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

            limiarizado = limiarizar(img)
            dilatado = dilatar(limiarizado, 1)

            # segmentacao de linha
            contorno, _ = cv2.findContours(
                dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            linhas_contorno = sorted(
                contorno, key=lambda ctr: cv2.boundingRect(ctr)[1]
            )  # x,y,w,h

            for ctr in linhas_contorno:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # segmentacao de palavra

            # segmentacao de caractere

            return img

        case "MobPhoto_1":
            pass
        case "MobPhoto_5":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = corrigirPerspectiva(img, arquivo)
            img = redimensionar(img, 4)
            img = limiarizar(img)
            img = dilatar(img, 1)
            return img

    # img = nitidizar(img)
    # img = blur(img, 2)
    # img = arestas(img, 105, 127)
