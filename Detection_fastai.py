import cv2
from fastai.basics import load_learner, os

learn = load_learner("mask.pkl")
GR_dict = {'without_mask': (0, 0, 255), 'with_mask': (0, 255, 0)}
rect_size = 4
cap = cv2.VideoCapture("http://192.168.1.27:8080/video")

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

haarcascade = cv2.CascadeClassifier(haar_model)

if haarcascade.empty():
    print("Cannot load!")
    exit()

while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1)

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y + h, x:x + w]
        rerect_sized = cv2.resize(face_img, (150, 150))
        # normalized = rerect_sized / 255.0
        # reshaped = np.reshape(normalized, (1, 150, 150, 3))
        # reshaped = np.vstack([reshaped])

        result = learn.predict(rerect_sized)
        label = result[0]

        cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)

        cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)

    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
