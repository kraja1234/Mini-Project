import sys

import cv2
import numpy as np
import face_recognition
import os
from  datetime import datetime

path='sample'

images = []
classnames =[]
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])
    print(classnames)

def findEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        try:
          encode=face_recognition.face_encodings(img)[0]
        except   IndexError as e:
          print(e)
          sys.exit(1)

        encodelist.append(encode)
    return encodelist

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDatalist=f.readlines()
        nameList=[]
        for line in myDatalist:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')






encodelistknown = findEncodings(images)
print('encoding complete')


#cap=cv2.VideoCapture(0)
cap=face_recognition.load_image_file('input/input.jpeg')

while True:
    #success ,img =cap.read()
    #imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(cap,cv2.COLOR_BGR2RGB)
    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodelistknown,encodeFace)
        faceDis=face_recognition.face_distance(encodelistknown,encodeFace)
        matchIndex=np.argmin(faceDis)


        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            markAttendance(name)



