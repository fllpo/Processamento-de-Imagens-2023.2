import cv2
from utils import input_file, output_file
import numpy as np
import pytesseract

img = cv2.imread(input_file("IMG_0122.jpg"),cv2.IMREAD_GRAYSCALE) 
    
def blur(img,metodo):
    match metodo:
        case 1:
            return cv2.blur(img, ksize=(9,9))
        case 2:
            return cv2.GaussianBlur(img, ksize=(9,9),sigmaX=30,sigmaY=30)
        case 3: 
            return cv2.medianBlur(img, 5)

def resize(img,size):
    return cv2.resize(img,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)

def arestas(img, t1,t2):
    return cv2.Canny(img,t1,t2)

def dilatar(img):
    k = np.ones((5,5),np.uint8)
    return cv2.dilate(img,k,iterations=1)

def erodir(img):
    k = np.ones((5,5),np.uint8)
    return cv2.erode(img,k,iterations=1)

def opening(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def removerRuido(img):
    return cv2.medianBlur(img,5)

def limiarizar(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template(img, template):
    return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) 

def preProcessamento(img):
    #img = removerRuido(img)
    img = limiarizar(img)
    img = dilatar(img)
    img = erodir(img)
    #img = opening(img)
    img = arestas(img,105,127)
    img = resize(img,1/2)
    return img

if __name__ == '__main__':
    
    preProcessamento(img)
    cv2.imshow('img',preProcessamento(img))

    custom_config = r'--oem 3 --psm 6'
    print(pytesseract.image_to_string(img, config=custom_config))

    cv2.waitKey(0)
    cv2.destroyAllWindows()