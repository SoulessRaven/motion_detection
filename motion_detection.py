import cv2
import time
import os

ADAPTIVE_BACKGROUND = False
CAPTURE_STATIC_BACKGROUND = False
timer = 10

video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)

avg_background = None
if not ADAPTIVE_BACKGROUND and not CAPTURE_STATIC_BACKGROUND:
    still_background_path = os.path.join("files", "background_gray.jpg")
    if not os.path.exists(still_background_path):
        print("[ERROR] Static background not found. Set CAPTURE_STATIC_BACKGROUND = True to create one.")
        exit()
    still_background = cv2.imread(still_background_path, 0)

if CAPTURE_STATIC_BACKGROUND == True:
    start_time = time.time()

while True:
    check, frame = video.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ADAPTIVE_BACKGROUND == True:
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg_background is None:
            avg_background = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg_background, 0.01)
        background_frame = cv2.convertScaleAbs(avg_background)

        delta = cv2.absdiff(background_frame, gray)
    else:
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray_still = cv2.GaussianBlur(still_background, (21, 21), 0)

        delta = cv2.absdiff(gray_still, gray)

    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)

    if CAPTURE_STATIC_BACKGROUND == True:
        if time.time() - start_time > timer:
            cv2.imwrite("background_gray.jpg", gray)
            break
    
    
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()