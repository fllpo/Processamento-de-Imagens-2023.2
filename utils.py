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


def log(PPimg, arquivo):
    extract = pytesseract.image_to_string(PPimg, config=custom_config)
    dir_saida = os.path.join("output/" + arquivo + "/log.txt")
    f = open(dir_saida, "w")
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


def redimensionaMostrar(img): #OK
    h, w, c = img.shape
    if w > 500:
        novo_w = 500
        ar = w / h
        novo_h = int(novo_w / ar)
        img = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    return img


def segmentacaoTexto(original): #OK
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    limiarizado = limiarizar(img)

    dilatado = cv2.dilate(limiarizado, np.ones((35, 35), np.uint8), iterations=1)

    contorno, _ = cv2.findContours(
        dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    secao_contorno_ordenada = sorted(contorno, key=lambda ctr: cv2.boundingRect(ctr)[1])

    PPimg_texto = original.copy()
    lista_texto = []

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        if (w or h) >= 50:
            lista_texto.append([x, y, x + w, y + h])
            cv2.rectangle(PPimg_texto, (x, y), (x + w, y + h), (100, 255, 255), 2)

    for i in range(len(lista_texto)):
        texto = lista_texto[i]
        saida = img[texto[1] : texto[3], texto[0] : texto[2]]
        dir_saida = os.path.join(
            "output/" + str(sys.argv[1]) + "/texto", "texto_" + str(i) + ".jpg"
        )
        cv2.imwrite(dir_saida, saida)

    return lista_texto


def segmentacaoLinha(original, lista_texto): #OK
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    lista_linha = []
    for i in range(len(lista_texto)):
        texto = lista_texto[i]
        PPimg_texto = img[texto[1] : texto[3], texto[0] : texto[2]]

        limiarizado = limiarizar(PPimg_texto)
        dilatado = cv2.dilate(limiarizado, np.ones((5, 39), np.uint8), iterations=1)

        contorno, _ = cv2.findContours(
            dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        secao_contorno_ordenada = sorted(
            contorno, key=lambda ctr: cv2.boundingRect(ctr)[1]
        )  # x,y,w,h

        PPimg_linha = cv2.cvtColor(PPimg_texto, cv2.COLOR_GRAY2BGR).copy()
        for ctr in secao_contorno_ordenada:
            x, y, w, h = cv2.boundingRect(ctr)
            lista_linha.append([x, y, x + w, y + h])
            cv2.rectangle(PPimg_linha, (x, y), (x + w, y + h), (100, 255, 100), 2)

            linha = lista_linha[len(lista_linha) - 1]
            saida = PPimg_texto[linha[1] : linha[3], linha[0] : linha[2]]

            dir_saida = os.path.join("output/" + str(sys.argv[1]) + "/linha", "linha_" + str(len(lista_linha) - 1) + ".jpg")
            cv2.imwrite(dir_saida, saida)

    return lista_linha


def segmentacaoPalavra(original):  # FIX
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    limiarizado = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[        1    ]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilatado = cv2.dilate(limiarizado, k, iterations=1)

    contorno, _ = cv2.findContours(        dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE    )
    secao_contorno_ordenada = sorted(contorno, key=lambda ctr: cv2.boundingRect(ctr)[1])

    PPimg_palavra = original.copy()
    lista_palavras = []

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        lista_palavras.append([x, y, x + w, y + h])
        cv2.rectangle(PPimg_palavra, (x, y), (x + w, y + h), (255, 255, 100), 2)

    # cv2.imshow("segmentacao de palavras", redimensionaMostrar(PPimg_palavra))

    for p in range(5):
        palavra = lista_palavras[p]
        PPimg_palavra = img[palavra[1] : palavra[3], palavra[0] : palavra[2]]
        # extract = pytesseract.image_to_string(PPimg_palavra, config=custom_config)
        dir_saida = os.path.join(
            "output/" + str(sys.argv[1]) + "/palavra", "palavra_" + str(p) + ".jpg"
        )
        cv2.imwrite(dir_saida, PPimg_palavra)

    return lista_palavras


def segmentacaoCaractere(original, lista_palavras):  # TODO
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    palavra = lista_palavras[2]
    img = img[palavra[1] : palavra[3], palavra[0] : palavra[2]]

    limiarizado = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    saida = cv2.erode(limiarizado, k, iterations=2)
    saida = cv2.dilate(saida, k)

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

    # cv2.imshow("caractere_" + str(i), PPimg_caractere)
    # extract = pytesseract.image_to_string(PPimg_caractere, config=custom_config)
    # print(str(i) + ": " + extract)


def preProcessamento(img, arquivo):
    match arquivo:
        case "IMG_0122":
            segTexto = segmentacaoTexto(img)
            segLinha = segmentacaoLinha(img, segTexto)
            # segPalavra = segmentacaoPalavra(img)
            # segCaractere = segmentacaoCaractere(img, segPalavra)

        case "MobPhoto_5":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = corrigirPerspectiva(img, arquivo)
            segTexto = segmentacaoTexto(img)
