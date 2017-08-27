# import lib of openCV
import cv2

#import the haarcascade frontl face 
classifier =cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

#import the haarcascade eye face
eyeClassifier= cv2.CascadeClassifier('cascades\haarcascade_eye.xml')

#import the haarcascede smile face
smileClassifier = cv2.CascadeClassifier('cascades\haarcascade_smile.xml')

#import photo for analyse 
imagem = cv2.imread('pessoas\\pessoas2.jpg')

#transform the photo in grayscale
imagemCinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)

#detect the frontal face
faceDetectadas =  classifier.detectMultiScale(imagemCinza)
#print the matrix of localization of image data
#print(faceDetectadas)

#loop to go through face
for(x, y, l , a ) in faceDetectadas:
    #print(x, y, l , a )

    # rectangle of information of frontal face
    cv2.rectangle(imagem,(x,y),(x + l, y +a),(0,0,255),2)

    #region of capture eye in frontal face
    regionEye = imagem[y:y+a, x:x+l]

    #region of capture smile in frontal face
    regionSmile = imagem[y:y+a, x:x+l]

    #print(regiao)

    #transform to eye place in grayscale
    regioGrayEye = cv2.cvtColor(regionEye,cv2.COLOR_BGR2GRAY)

    #transform the image in grayscale to capture smile by face
    regionGraySmile = cv2.cvtColor(regionSmile, cv2.COLOR_BGR2GRAY)

    #detected the eye in image
    detectedEye = eyeClassifier.detectMultiScale(regioGrayEye, scaleFactor = 1.01, minNeighbors = 1)

    #print(detectedEye)
    #test of smile capture
    smileDetected =  smileClassifier.detectMultiScale(regionGraySmile, scaleFactor= 1.11, minNeighbors=8)

    #print(len(smileDetected))

	#loop to go through eye face
    for(ox, oy, ol, oa) in detectedEye:
        #print(ox, oy, ol , oa )
        cv2.rectangle(regionEye, (ox,oy),(ox + ol, oy + oa), (0,255,0),2)

    #loop to go through smile face
    for(sx, sy , sl ,sa) in smileDetected:
    	#print(sx, sy, sl, sa)
    	cv2.rectangle(regionSmile, (sx,sy),(sx + sl, sy + sa), (0,255,0),2)

#window show the colect data    
cv2.imshow("Faces, Eyes and Smile", imagem)

#wait the key to close window
cv2.waitKey()
