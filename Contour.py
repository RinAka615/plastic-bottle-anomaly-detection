import cv2 as cv
import numpy as np
import imutils

image = cv.imread("Figure_3.png")
image = imutils.resize(image, width=600)
cv.imshow("Original Image", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)[1]
cv.imshow("Thresholded Image", thresh)

cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

c = max(cnts, key=cv.contourArea)
output = image.copy()
cv.drawContours(output, [c], -1, (0, 255, 0), 2)
(x, y, w, h) = cv.boundingRect(c)
text = "original, N={}".format(len(c))
cv.putText(output, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print("Information: {}", format(text))
cv.imshow("Original Outline", output)

for eps in np.linspace(0.001, 0.05, 10):
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, eps * peri, True)
    output = image.copy()
    cv.drawContours(output, [approx], -1, (0, 255, 0), 2)
    text = "eps={:.3f}, N={}".format(eps, len(approx))
    cv.putText(output, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("Information: {}".format(text))
    cv.imshow("Approximation", output)
    cv.waitKey(0)
cv.waitKey(0)
cv.destroyAllWindows()