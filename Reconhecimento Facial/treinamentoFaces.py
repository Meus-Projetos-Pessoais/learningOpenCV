import cv2
import numpy as np
import os

eigenface =  cv2.face.EigenFaceRecognizer_create(num_components=50,threshold=0.05)

fisherface =  cv2.face.FisherFaceRecognizer_create(num_components=50)

lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():

    caminhos =  [os.path.join('facesTreinamento', f) for f in os.listdir('facesTreinamento')]


    #print(caminhos)

    faces = []
    ids = []

    for caminhoImagem in caminhos:
        imagemFace =  cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #print(id)
        ids.append(id)

        faces.append(imagemFace)

        #cv2.imshow("Faces", imagemFace)
        #cv2.waitKey(10)
    return np.array(ids),faces

ids , faces = getImagemComId()

#print(ids)
#print(faces)


print("Treinando ... ")


eigenface.train(faces,ids)
eigenface.write('ArquivosTreinamentos\classificadorEigen.yml')

fisherface.train(faces,ids)
fisherface.write('ArquivosTreinamentos\classificadorFisher.yml')

lbph.train(faces,ids)
lbph.write('ArquivosTreinamentos\classificadorLBPH.yml')

print("Treinamento realizado.")