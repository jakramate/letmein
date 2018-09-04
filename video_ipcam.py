# various libraries
import cv2                   # opencv
import urllib.request        # handling reading webcam data
import numpy as np           # numpy
from sklearn import mixture  # mixture model
from os import listdir

debug = False

def prepare_training_data():
    faces = []
    annotations = []

    labels = listdir()

    for label in labels:
        if 'py' not in label:
            images = listdir(label)

    i = 0
    for img in images:
        im = cv2.imread(label + '/' + img)
        im = cv2.resize(im, (100, 100), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces.append(gray) 
        annotations.append(i)
        i += 1 
        
    return (faces,annotations)


# initialising face recogniser
faces, labels = prepare_training_data()
face_recogniser = cv2.face.LBPHFaceRecognizer_create()
face_recogniser.train(faces, np.array(labels))  # training the recogniser

subject = ['','Jo','Sherbet']
with urllib.request.urlopen('http://172.16.10.103/video.cgi') as src:
#with urllib.request.urlopen('http://172.16.10.103/IMAGE.JPG?cidx=2018831847382687') as src:

    byteString = b'' 

    i = 0
    while True:
        # reading data from http
        byteString += src.read(1024)

        # extracting video frame (which is actually a series of jpeg format)
        a = byteString.find(b'\xff\xd8')
        b = byteString.find(b'\xff\xd9')
        
        if a > 0 and b > 0:
            print('hasdfasdfas')
            jpg = byteString[a:b+2]
            byteString = byteString[b+2:]            

            # fabricate the video frame and stored as an image in memory    
            im = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
            eye_icascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Gray', gray)

            #try:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            #print(len(faces))

            if debug:
                print(len(faces))        

        
            for (x,y,w,h) in faces:
                cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = im[y:y+h, x:x+w]
                #eyes = eye_cascade.detectMultiScale(roi_gray)            
                #for (ex,ey,ew,eh) in eyes:
                #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

                # roi_color is the cropped image
                crop_im = cv2.resize(roi_gray, (100, 100), interpolation = cv2.INTER_CUBIC)
                label,value = face_recogniser.predict(crop_im)
                name = subject[label]

                #if not grant[name]:
                # display welcome and grant access
                #else:
                # putting name on roi
                cv2.putText(im, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                # now we need to attach name to
                #if i % 100 == 1:
                #    picname = str(i) + '.jpg'
                #    cv2.imwrite(picname, roi_color)

                #i = i + 1
            
            if cv2.waitKey(25) == 27:
                exit(0)
