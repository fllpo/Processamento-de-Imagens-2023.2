from utils import *

arquivo = sys.argv[1]

img = cv2.imread(input_file(arquivo + ".jpg"))
PPimg = preProcessamento(img, arquivo)

# print(pytesseract.image_to_string(PPimg, config=custom_config))
log(PPimg, arquivo)
mostrar(img, PPimg, arquivo)

cv2.waitKey(0)
cv2.destroyAllWindows()
