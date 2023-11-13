import os
import sys
import cv2
import numpy as np
import pytesseract

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
custom_config = r"--oem 3 --psm 10"


def input_file(filename):
    return os.path.join(INPUT_FOLDER, filename)


def output_file(filename):
    return os.path.join(OUTPUT_FOLDER, filename)


def mostrar(img, PPimg, arquivo):
    match arquivo:
        case "IMG_0122":
            cv2.imshow("PPimg", PPimg)
            cv2.imshow("Original", img)

            pass
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


def limiarizar(img):
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


def redimensionaMostrar(img):
    h, w, c = img.shape
    if w > 500:
        novo_w = 500
        ar = w / h
        novo_h = int(novo_w / ar)
        img = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    return img


def localizacaoTexto(original):
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    limiarizado = limiarizar(img)

    dilatado = cv2.dilate(limiarizado, np.ones((35, 35), np.uint8), iterations=1)

    contorno, _ = cv2.findContours(
        dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    secao_contorno_ordenada = sorted(contorno, key=lambda ctr: cv2.boundingRect(ctr)[1])

    PPimg_texto = original.copy()
    lista_textos = []

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        lista_textos.append([x, y, x + w, y + h])
        cv2.rectangle(PPimg_texto, (x, y), (x + w, y + h), (100, 255, 100), 2)

    PPimg_texto = redimensionaMostrar(PPimg_texto)

    for i in range(len(lista_textos)):
        texto = lista_textos[i]
        PPimg_texto = img[texto[1] : texto[3], texto[0] : texto[2]]

        return lista_textos
        # cv2.imshow("texto "+str(i), PPimg_texto)


def segmentacaoLinha(original):
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    limiarizado = limiarizar(img)
    dilatado = cv2.dilate(limiarizado, np.ones((5, 39), np.uint8), iterations=1)

    contorno, _ = cv2.findContours(
        dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    secao_contorno_ordenada = sorted(
        contorno, key=lambda ctr: cv2.boundingRect(ctr)[1]
    )  # x,y,w,h
    PPimg_linha = original.copy()

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(PPimg_linha, (x, y), (x + w, y + h), (100, 255, 100), 2)

    PPimg_linha = redimensionaMostrar(PPimg_linha)

    # cv2.imshow("segmentacao de linhas", PPimg_linha)


def segmentacaoPalavra(original):  # OK
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    limiarizado = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
        1
    ]
    
    k=cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  
    dilatado = cv2.dilate(limiarizado, k, iterations=1)

    contorno, _ = cv2.findContours(
        dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    secao_contorno_ordenada = sorted(contorno, key=lambda ctr: cv2.boundingRect(ctr)[1])

    PPimg_palavra = original.copy()
    lista_palavras = []

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        lista_palavras.append([x, y, x + w, y + h])
        cv2.rectangle(PPimg_palavra, (x, y), (x + w, y + h), (255, 255, 100), 2)

    return lista_palavras


def segmentacaoCaractere(original, lista_palavras):  # TODO
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    palavra = lista_palavras[2]
    img = img[palavra[1] : palavra[3], palavra[0] : palavra[2]]

    limiarizado = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    k=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    
    saida = cv2.erode(limiarizado, k, iterations=2)
    saida = cv2.dilate(saida, k)

    cv2.imshow("saida", saida)

    contorno, _ = cv2.findContours(
        saida.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    secao_contorno_ordenada = sorted(contorno, key=lambda ctr: cv2.boundingRect(ctr)[1])

    PPimg_caractere = saida.copy()
    lista_caracteres = []

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        lista_caracteres.append([x, y, x + w, y + h])
        cv2.rectangle(PPimg_caractere, (x, y), (x + w, y + h), (100, 100, 255), 1)

    for i in range(len(lista_caracteres)):
        caractere = lista_caracteres[i]
        PPimg_caractere = saida[
            caractere[1] : caractere[3], caractere[0] : caractere[2]
        ]
        
        cv2.imshow("caractere_" + str(i), PPimg_caractere)
        #extract = pytesseract.image_to_string(PPimg_caractere, config=custom_config)
        #print(str(i) + ": " + extract)


def preProcessamento(img, arquivo):
    match arquivo:
        case "IMG_0122":
            teste = localizacaoTexto(img)
            # segmentacaoLinha(img)
            # segmentacaoPalavra(img)
            segmentacaoCaractere(img, segmentacaoPalavra(img))

            # return PPimg_palavra

        case "MobPhoto_1":
            pass
        case "MobPhoto_5":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = corrigirPerspectiva(img, arquivo)
            img = redimensionar(img, 4)
            img = limiarizar(img)
            img = dilatar(img, 1)
            return img
