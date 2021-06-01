import cv2
import sys
from llist import llist

# Command Lineda image va cascade fileni olish
imagePath = sys.argv[1]
cascadePath = sys.argv[2]


# Haar cascade
faceCascade = cv2.CascadeClassifier(cascadePath)


# imageni o'qish
image = cv2.imread(imagePath)
# imageni grayscalega convert qilish
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
)


facesCount = len(faces)

print(f"Found {facesCount} faces")


for (x, y, w, h) in faces:
    oneFace = {'x': x, 'y': y, 'width': w, 'height': h}
    llist.append(oneFace)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


print(llist.printList())

cv2.imshow("Faces found", image)
cv2.waitKey(0)
