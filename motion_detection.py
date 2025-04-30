import cv2

video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)

while True:
    check, frame = video.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)

    cv2.imshow("Capturing", frame)


    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()