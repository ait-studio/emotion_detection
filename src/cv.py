from fer import FER
import cv2

def emotionDetecte(filename):
    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    emotion_detector = FER(mtcnn=True)

    filePath = "./uploads/"
    img = cv2.imread(filePath + filename)
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

    analysis = emotion_detector.detect_emotions(img)
    if len(analysis) > 0 and len(faces) > 0:
        gap = 10
        fullWidth = 100
        fullHeight = 20
        x = faces[0][0]
        y = faces[0][1]
        for i in range(7):
            target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][i]
            
            startPoint = (x + w + gap, y + i * (gap + fullHeight))

            endPointX = x + w + gap + fullWidth
            partialEndX = x + w + gap + int(fullWidth * analysis[0]['emotions'][target])

            endPointY = y + fullHeight + i * (gap + fullHeight)

            endPoint = (endPointX, endPointY)
            partialEnd = (partialEndX, endPointY)
            cv2.rectangle(img, startPoint, endPoint, (125, 125, 125), -1)
            cv2.rectangle(img, startPoint, partialEnd, (0, 0, 255), -1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            fontPosition = (x + w + gap, y + i * (gap + fullHeight) + 20)
            cv2.putText(img, target, fontPosition, font, 1, color, 2)
    else: 
        position = (10, 30)
        rectStPoint = (5, 3)
        rectEndPoint = (303, 38)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)
        cv2.rectangle(img, rectStPoint, rectEndPoint, (0, 0, 0), -1)
        cv2.putText(img, "No face found", position, font, 1.2, color, 3)
    
    saveFilename = filename.split(".png")[0] + "_analyzed.png"
    saveFilepath = "./analyzed/"
    cv2.imwrite(saveFilepath + saveFilename, img)

    aws_link_header = "https://parkinsense.s3.ap-northeast-2.amazonaws.com/"

    if(len(analysis) > 0): 
        return {"status": "200 ok", "result": analysis[0]['emotions'], "link": aws_link_header + saveFilename}
    else:
        return {"status": "200 ok", "result": "no face or emotion detected", "link": aws_link_header + saveFilename}
