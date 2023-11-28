from utils import *


def segmentacaoLinha(original, lista_texto):  # OK
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

            dir_saida = os.path.join(
                "output/"
                + str(sys.argv[1])
                + "/linha/linha_"
                + str(len(lista_linha) - 1)
                + ".jpg"
            )
            cv2.imwrite(dir_saida, saida)
