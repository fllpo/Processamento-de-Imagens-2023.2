import os
import sys
import cv2
import numpy as np
import pytesseract


INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
custom_config = r"--oem 3 --psm 8"


def input_file(filename):
    return os.path.join(INPUT_FOLDER, filename)


def output_file(filename):
    return os.path.join(OUTPUT_FOLDER, filename)


def nitidizar(img):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    return cv2.filter2D(img, -1, k)


def corrigirPerspectiva(img, arquivo):
    match arquivo:
        case "IMG_0122":
            pass
        case "MobPhoto_5":
            pts1 = np.float32([[280, 180], [1625, 210], [47, 1520], [1860, 1520]])
            pts2 = np.float32([[0, 0], [1536, 0], [0, 2048], [1536, 2048]])
            matriz = cv2.getPerspectiveTransform(pts1, pts2)

            return cv2.warpPerspective(img, matriz, (1536, 2048))


def redimensionar(img):  # OK
    h, w, c = img.shape
    if w > 500:
        novo_w = 500
        ar = w / h
        novo_h = int(novo_w / ar)
        img = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    return img


def segmentacaoTexto(original):
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    limiarizado = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
        1
    ]

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

            PPimg_texto = cv2.putText(
                PPimg_texto,
                str(len(lista_texto)),
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            PPimg_texto = cv2.putText(
                PPimg_texto,
                str(len(lista_texto)),
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            texto = lista_texto[len(lista_texto) - 1]
            saida = img[texto[1] : texto[3], texto[0] : texto[2]]
            dir_saida = os.path.join(
                "output/" + str(sys.argv[1]) + "/texto",
                "texto_" + str(len(lista_texto) - 1) + ".jpg",
            )
            cv2.imwrite(dir_saida, saida)

    #cv2.imshow("segmentacao de texto", redimensionar(PPimg_texto))
    dir_saida = os.path.join("output/" + str(sys.argv[1]) + "/segmentação de texto.jpg")
    cv2.imwrite(dir_saida, PPimg_texto)


def segmentacaoPalavra(original):
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    lista_palavra_seg = []

    limiarizado = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
        1
    ]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilatado = cv2.dilate(limiarizado, k, iterations=1)

    contorno, _ = cv2.findContours(
        dilatado.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    secao_contorno_ordenada = sorted(contorno, key=lambda ctr: cv2.boundingRect(ctr)[1])

    PPimg_palavra = original.copy()

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        if w > 10 and h > 10:
            lista_palavra_seg.append([x, y, x + w, y + h])
            cv2.rectangle(PPimg_palavra, (x, y), (x + w, y + h), (255, 255, 100), 2)

            palavra = lista_palavra_seg[len(lista_palavra_seg) - 1]
            saida = cv2.cvtColor(
                original[palavra[1] : palavra[3], palavra[0] : palavra[2]],
                cv2.COLOR_BGR2GRAY,
            )

            dir_saida = os.path.join(
                "output/"
                + str(sys.argv[1])
                + "/palavra/palavra_"
                + str(len(lista_palavra_seg) - 1)
                + ".jpg"
            )
            cv2.imwrite(dir_saida, saida)

            extract = pytesseract.image_to_string(saida, config=custom_config)
            extract = extract[: len(extract) - 2]

            PPimg_palavra = cv2.putText(
                PPimg_palavra,
                extract,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            PPimg_palavra = cv2.putText(
                PPimg_palavra,
                extract,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    dir_saida = os.path.join(
        "output/" + str(sys.argv[1]) + "/segmentação de palavras.jpg"
    )
    cv2.imwrite(dir_saida, PPimg_palavra)
    #cv2.imshow("segmentacao de palavras", redimensionar(PPimg_palavra))


def segmentacaoCaractere(original):  # TODO
    custom_config = r"--oem 3 --psm 10"

    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    lista_palavra_seg = []

    limiarizado = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
        1
    ]

    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    opening = cv2.morphologyEx(limiarizado, cv2.MORPH_OPEN, k)

    contorno, _ = cv2.findContours(
        opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    secao_contorno_ordenada = sorted(contorno, key=lambda ctr: cv2.boundingRect(ctr)[1])

    PPimg_palavra = original.copy()

    for ctr in secao_contorno_ordenada:
        x, y, w, h = cv2.boundingRect(ctr)
        if w > 10 and h > 10:
            lista_palavra_seg.append([x, y, x + w, y + h])
            cv2.rectangle(PPimg_palavra, (x, y), (x + w, y + h), (100, 100, 255), 2)

            palavra = lista_palavra_seg[len(lista_palavra_seg) - 1]
            saida = cv2.cvtColor(
                original[palavra[1] : palavra[3], palavra[0] : palavra[2]],
                cv2.COLOR_BGR2GRAY,
            )

            saida = cv2.threshold(
                saida, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]

            dir_saida = os.path.join(
                "output/"
                + str(sys.argv[1])
                + "/caractere/caractere_"
                + str(len(lista_palavra_seg) - 1)
                + ".jpg"
            )
            cv2.imwrite(dir_saida, saida)

            extract = pytesseract.image_to_string(saida, config=custom_config)
            extract = extract[: len(extract) - 2]

            PPimg_palavra = cv2.putText(
                PPimg_palavra,
                extract,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            PPimg_palavra = cv2.putText(
                PPimg_palavra,
                extract,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    dir_saida = os.path.join(
        "output/" + str(sys.argv[1]) + "/segmentação de caracteres.jpg"
    )
    cv2.imwrite(dir_saida, PPimg_palavra)
    #cv2.imshow("segmentacao de caractere", redimensionar(PPimg_palavra))


def Processamento(img, arquivo):
    match arquivo:
        case "IMG_0122":
            segmentacaoTexto(img)
            segmentacaoPalavra(img)
            segmentacaoCaractere(img)

        case "MobPhoto_5":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = corrigirPerspectiva(img, arquivo)
            segmentacaoTexto(img)
            segmentacaoPalavra(img)
            segmentacaoCaractere(img)
