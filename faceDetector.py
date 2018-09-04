# @Author: Jakramate Bootkrajang
# @Date: 06 Apr 2018

# various libraries
import cv2  # opencv
import numpy as np           
from os import listdir
from os import path 
from scipy import stats
class CMUFaceRecogniser(object):

    def __init__(self):
        self.recogniser1 = cv2.face.LBPHFaceRecognizer_create()
        self.recogniser2 = cv2.face.EigenFaceRecognizer_create()
        self.recogniser3 = cv2.face.FisherFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
        
        self.retrain()
        #faces, labels, self.annotations = self.prepare_training_data()
        #self.recogniser.train(faces, np.array(labels))  # training the recogniser
        #if len(faces) > 0:
        #    self.recogniser1.train(faces, np.array(labels))  # training the recogniser
        #    self.recogniser2.train(faces, np.array(labels))  # training the recogniser
        #    self.recogniser3.train(faces, np.array(labels))  # training the recogniser
        #else:
        #    print('No training data, collect the data first')

 #   def set_recogniser(self, newRecogniser):
 #       self.recogniser = newRecogniser

    def retrain(self):
        #self.recogniser = cv2.face.LBPHFaceRecognizer_create()
        #self.detector = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
        faces, labels, self.annotations = self.prepare_training_data()

        if len(faces) > 0:
            self.recogniser1.train(faces, np.array(labels))  # training the recogniser
            self.recogniser2.train(faces, np.array(labels))  # training the recogniser
            self.recogniser3.train(faces, np.array(labels))  # training the recogniser
        else:
            print('No training data, collect the data first')
            

    def prepare_training_data(self):
        faces = []
        labels = []
        annotations = []

        folders = listdir()

        i = 0
        for folder in folders:
            if path.isdir(folder) and '__' not in folder:
                images = listdir(folder)
                
                for img in images:
                    im = cv2.imread(folder + '/' + img)
                    im = cv2.resize(im, (100, 100), interpolation = cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces.append(gray) 
                    labels.append(i)
                
                annotations.append(folder)
                i = i + 1

        #print(labels)
        return (faces, labels, annotations)



    def detect_face(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, 0, (150,150), (300,300))

        return faces


    def recognise_face(self, im, faces):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        
            # send the resized roi_gray to the recogniser
            crop_im = cv2.resize(roi_gray, (100, 100), interpolation = cv2.INTER_CUBIC)
            label1, confident_value1 = self.recogniser1.predict(crop_im)
            label2, confident_value2 = self.recogniser2.predict(crop_im)
            label3, confident_value3 = self.recogniser3.predict(crop_im)
            label = stats.mode([label1, label2, label3])
            name = self.annotations[label.mode[0]]
            

            # put recognised name on roi's border
            #print(confident_value)
            if (confident_value1+confident_value2+confident_value3)/3 > 50:
                cv2.putText(im, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(im, 'Unknown', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
        return im   # im is now annotated 
