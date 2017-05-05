import cv2
import numpy as np 
import sqlite3

def _putText(img,output_string,point,fonts,line_type,color,thickness):
	'''
	Since opencv puttext does not accept multiple lines, custom function is needed here
	'''
	x = point[0]
	y0, dy = point[1], int(thickness*0.3+line_type*12)
	print(output_string.split('/n'))
	for i, text in enumerate(output_string.split('/n')):
		y = y0 + i*dy
		cv2.putText(img, text, (x,y), fonts, line_type, color,thickness)

conn = sqlite3.connect('database.db')

c = conn.cursor()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('recognizer/trainingData.yml')
ids = 0
font = cv2.FONT_HERSHEY_PLAIN   
while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
		c.execute("select name,position from users where id = (?);", (ids,))
		result = c.fetchall()
		name = result[0][0]
		position = result[0][1]
		if conf < 50:
			output_string = 'Name: '+name+'/nPosition: '+position
			_putText(img,output_string,(x,y+h), font, 1,(255,0,0),1)
		else:
			_putText(img,'Face not in the database',(x,y+h), font, 1,(0,0,255),1)
	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()