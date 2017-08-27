import cv2

print(cv2.__version__)

imagem = cv2.imread('opencv-python.jpg')
imagemCinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Cinza",imagemCinza)
cv2.imshow("Imagem original",imagem)
cv2.waitKey()
