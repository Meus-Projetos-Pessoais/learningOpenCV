import cv2
import numpy as np

classificadorFaces =  cv2.CascadeClassifier("cascades\haarcascade_frontalcatface.xml")

classificadorOlhos =  cv2.CascadeClassifier("cascades\haarcascade_eye.xml")


camera = cv2.VideoCapture(0)

amostra = 1
numeroAmostras = 25
id = input("Digite seu nome(Identificador) :")

largura, altura = 220 ,220

print("Capurando as faces ...")

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    #print(np.average(imagemCinza))
    facesDetectadas =  classificadorFaces.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150,150))


    for(x, y , l, a) in facesDetectadas:

        cv2.rectangle(imagem, (x , y),(x + l , y + a ),(0,0,255) ,2)

        regiaoOlhos = imagem[y:y + a , x:x + l ]

        regiaoOlhosCinza =  cv2.cvtColor(regiaoOlhos, cv2.COLOR_BGR2GRAY)

        olhosDetectados =  classificadorOlhos.detectMultiScale(regiaoOlhosCinza)

        for(ox, oy , ol, oa) in olhosDetectados:

            cv2.rectangle(regiaoOlhos, (ox, oy),(ox + ol, oy + oa), (0,255,0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                 if np.average(imagemCinza)>= 50:
                     imagemFace =  cv2.resize(imagemCinza[y:y +a , x:x + l], (largura, altura))
                     cv2.imwrite("facesTreinamento\pessoa_" + str(id) + "_" + str(amostra) + ".jpg", imagemFace)
                     print("[Foto " + str(amostra) + "capturada com sucesso]")
                     amostra =  amostra + 1



    cv2.imshow("Captura de faces", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break

print("Faces capturadas com sucesso")



