from utils import *

arquivo = sys.argv[1]

img = cv2.imread(input_file(arquivo + ".jpg"))
Processamento(img, arquivo)

cv2.waitKey(0)
cv2.destroyAllWindows()
