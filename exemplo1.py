import cv2

classificador =cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

imagem = cv2.imread('pessoas\\pessoas3.jpg')

imagemCinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)

faceDetectadas =  classificador.detectMultiScale(imagem, scaleFactor =1.11, minNeighbors = 9, minSize = (30,30))
print(len(faceDetectadas))
print(faceDetectadas)


for(x, y, l , a ) in faceDetectadas:
    print(x, y, l , a )
    cv2.rectangle(imagem,(x,y),(x + l, y +a),(0,55,255),2)

cv2.imshow("Faces encontradas", imagem)
cv2.waitkey()
