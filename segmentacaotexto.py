from utils import *


def segmentacaoTexto(original):
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

    cv2.imshow("segmentacao de texto", redimensionar(PPimg_texto))

    return lista_texto
