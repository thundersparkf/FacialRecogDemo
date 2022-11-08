# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file('images/emma_1.jpeg')
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)
face = face_recognition.face_locations(imgelon)[0]
copy = imgelon.copy()
cv2.rectangle(copy, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 2)

train_elon_encodings = face_recognition.face_encodings(imgelon)[0]
print(np.size(train_elon_encodings))

test = face_recognition.load_image_file('images/emma_2.jpeg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]
print(face_recognition.compare_faces([train_elon_encodings], test_encode))

test_2 = face_recognition.load_image_file('images/zuck_1.jpeg')
test_2 = cv2.cvtColor(test_2, cv2.COLOR_BGR2RGB)
test_2_encode = face_recognition.face_encodings(test_2)[0]
#print(face_recognition.compare_faces([train_elon_encodings], test_2_encode))

cv2.imshow('copy', copy)
cv2.imshow('elon', imgelon)
cv2.waitKey(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
