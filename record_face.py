import cv2
import numpy as np 
import sqlite3

conn = sqlite3.connect('database.db')

c = conn.cursor()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

uname = input("Enter your name: ")
spirit_animal = input("Enter your spirit animal: ")

c.execute('INSERT INTO users (name,spirit_animal) VALUES (?,?)', (uname,spirit_animal))

uid = c.lastrowid

sampleNum = 0

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		sampleNum = sampleNum+1
		cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
		cv2.waitKey(100)
	cv2.imshow('img',img)
	cv2.waitKey(1);
	if sampleNum > 30:
		break
cap.release()

conn.commit()

conn.close()
cv2.destroyAllWindows()