import cv2
import time
import os
import pandas as pd
from datetime import datetime as dt

ADAPTIVE_BACKGROUND = False
CAPTURE_STATIC_BACKGROUND = False
timer = 10

video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)

status_list = [None, None]
times = []
log_df = pd.DataFrame(columns=["Start", "End"])


if CAPTURE_STATIC_BACKGROUND == True:
    start_time = time.time()

    while True:
        check, frame = video.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if time.time() - start_time > timer:
            bg_path = os.path.join("files", "background_gray.jpg")
            cv2.imwrite(bg_path, gray)
            print("[INFO] Background saved to", bg_path)
            break

        cv2.imshow("Capturing", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)

avg_background = None
if not ADAPTIVE_BACKGROUND:
    still_background_path = os.path.join("files", "background_gray.jpg")
    if not os.path.exists(still_background_path):
        print("[ERROR] Static background not found. Set CAPTURE_STATIC_BACKGROUND = True to create one.")
        exit()
    still_background = cv2.imread(still_background_path, 0)

while True:
    check, frame = video.read()
    status = 0
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

    threshold = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None, iterations = 3)

    (cnts,_) = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1500:
            continue
        
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(dt.now())
        md_base_name = dt.now().strftime("%Y%m%d_%H%M%S")
        md_output_name = f"{md_base_name}_detected.jpg"
        os.makedirs("LOG", exist_ok=True)
        log_jpg = os.path.join("LOG", md_output_name)
        cv2.imwrite(log_jpg, frame)

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(dt.now())

    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)
 
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

if status == 1:
    times.append(dt.now())

os.makedirs("LOG", exist_ok=True)
log_path = os.path.join("LOG", "detection_times.csv")

for i in range(o, len(times), 2):
    log_df = log_df._append({"Start": times[1].isoformat(), "End": times[i + 1].isoformat()}, ignore_index = True)

log_df.to_csv(log_path, index_label = "Index")