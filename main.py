from fer import FER
import cv2

cap = cv2.VideoCapture(0)

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

emotion_detector = FER(mtcnn=True)

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if ret == True :
        cv2.imshow("img", img)
        analysis = emotion_detector.detect_emotions(img)
        print(analysis)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()

