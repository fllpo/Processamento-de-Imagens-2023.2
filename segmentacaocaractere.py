from utils import *


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
