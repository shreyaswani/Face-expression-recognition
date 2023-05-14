import cv2
import matplotlib.pyplot as plt
from fer import FER

cap = cv2.VideoCapture(0)
emotion_detector = FER(mtcnn=True)

while True:
    ret, frame = cap.read()
    result = emotion_detector.detect_emotions(frame)
    if result:
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]
        cv2.rectangle(frame, (
            bounding_box[0], bounding_box[1]), (
                          bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255), 2, )

        for index, (emotion_name, score) in enumerate(emotions.items()):
            color = (211, 211, 211) if score < 0.01 else (0, 0, 255)
            emotion_score = "{}: {:.2f}".format(emotion_name, score)
            cv2.putText(frame, emotion_score,
                        (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA, )
    flipped = cv2.flip(frame, 1)
    cv2.imshow('frame',flipped)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        cv2.imwrite('emotion.jpg', frame)
        img_sh = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img_sh)
        plt.axis("off")
        plt.show()
        cv2.destroyAllWindow()

cap.release()
cv2.destroyAllWindows()
