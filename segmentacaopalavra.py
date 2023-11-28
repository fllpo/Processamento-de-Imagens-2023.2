from utils import *

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
custom_config = r"--oem 3 --psm 8"


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
    # cv2.imshow("segmentacao de palavras", redimensionar(PPimg_palavra))

    return lista_palavra_seg
