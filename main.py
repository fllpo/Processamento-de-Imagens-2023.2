from utils import *

arquivo = "IMG_0122"

img = cv2.imread(input_file(arquivo + ".jpg"), cv2.IMREAD_GRAYSCALE)
PPimg = preProcessamento(img)

log(arquivo, PPimg)
print(pytesseract.image_to_string(PPimg, config=custom_config))

mostrar(img, PPimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
