import cv2 as cv
import numpy as np
from pyk4a import PyK4A
import sys
import os

SAVE_DIR = "image"

k4a = PyK4A()
k4a.start()

capture = k4a.get_capture()

if capture.color is None:
    k4a.stop()
    sys.exit()

first_frame = capture.color[:, :, :3]
first_frame = np.ascontiguousarray(first_frame)

#roi = cv.selectROI("Keo chuot chon vung roi bam ENTER", first_frame, showCrosshair=True, fromCenter=False)
# cv.destroyWindow("Keo chuot chon vung roi bam ENTER")
# (x, y, w, h) = roi

(x, y, w, h) = (494, 116, 347, 596)

if w > 0 and h > 0:
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    img_count = 0
    for file_name in os.listdir(SAVE_DIR):
        if file_name.startswith("crop_") and file_name.endswith(".png"):
            try:
                num = int(file_name.split("_")[1].split(".")[0])
                if num > img_count:
                    img_count = num
            except ValueError:
                continue
    
    while True:
        cap = k4a.get_capture()
        
        if cap.color is not None:
            frame = cap.color[:, :, :3]
            frame = np.ascontiguousarray(frame)
            
            cropped_frame = frame[y : y+h, x : x+w].copy() 
            
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.imshow("Video Tong The", frame)
            
        key = cv.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == ord(' '):
            img_count += 1
            filename = os.path.join(SAVE_DIR, f"crop_{img_count}.png")
            cv.imwrite(filename, cropped_frame)

k4a.stop()
cv.destroyAllWindows()