import cv2
import numpy as np
from numpy.lib import imag
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time


X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
#print(pd.Series(y).value_counts())

classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)

#Splitting the data
xTrain,xTest,yTrain,yTest=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
xTrainScaled=xTrain/255.0
xTestScaled=xTest/255.0
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xTrainScaled,yTrain)
yPrediction=clf.predict(xTestScaled)
accuracy=accuracy_score(yTest,yPrediction)

print("Accuracy: ",accuracy)

#Using the camera for showing the digit
#Starting the camera
capture=cv2.VideoCapture(0)

while(True):
    #capturing frame by frame
    try:
        ret,frame= capture.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #drawing the box in the center of the video
        height,width=gray.shape
        upperLeft=(int(width/2-56),int(height/2-56)) 
        bottomRight=(int(width/2+56),int(height/2+56)) 
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0,2))
        #To consider the area inside the box for detecting the digit
        #roi=region of interest
        roi=gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]
        #converting to pill format
        imgPillow=Image.fromarray(roi)
        imageBw=imgPillow.convert("L")
        imageResize=imageBw.resize((28,28),Image.ANTIALIAS)
        imageInverted=PIL.ImageOps.invert(imageResize)
        pixelFilter=20
        minPixel=np.percentile(imageInverted,pixelFilter)
        imageScaled=np.clip(imageInverted-minPixel,0,255)
        maxPixel=np.max(imageInverted)
        imageScaled=np.asarray(imageScaled)/maxPixel
        testSample=np.array(imageScaled).reshape(1,784)
        prediction=clf.predict(testSample)
        print("Prediction is: ",prediction)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1)& 0xff==ord("q"):
            break

    except Exception as e:
        pass

capture.release()
cv2.destroyAllWindows() 