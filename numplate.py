import cv2
import pytesseract as pyt
import os

path=r'C:\Users\lenovo\PycharmProjects\ocv\Resources\carplaterecord' #change this to your numberplate output folder

pyt.pytesseract.tesseract_cmd = 'F:\\Program Files\\Tesseract-OCR\\tesseract.exe' #OPTIONAL tesseract location
nPlateCascade = cv2.CascadeClassifier('./Resources/haarcascade_russian_plate_number.xml') #location of numerplate xml file
count = 0
cap = cv2.VideoCapture(0)

while True:
    ret,frame =cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    numberplate=nPlateCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 5, minSize=(30,30))



    for x, y, w, h in numberplate:

        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 5)
        imgcrop=frame[y:y+h,x:x+w]

        count = count + 1
        fname = str(count) + ".jpg"
       # reader = pyt.image_to_string(imgcrop)
        cv2.imwrite(os.path.join(path,fname),imgcrop)

        cv2.imshow('cropped',imgcrop)
        print(count)


    cv2.imshow('Text', gray)


    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break