import numpy.core.multiarray
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import copy as cp

clf = joblib.load("digits_cls.pkl")

im = cv2.imread("photo_4.jpg")

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]

xs = []
ys = []
for i in rects:
    xs.append(i[0])
    ys.append(i[1])
xs_sorted = sorted(xs)
ys_sorted = sorted(ys)
least_x = xs_sorted[0]
least_y = ys_sorted[0]

ys_sorted_modified = []
for i in ys_sorted:
    if(i<=least_y+50 and i>least_y-50):
        ys_sorted_modified.append(least_y)
    else:
        least_y = i

ys_sorted_modified = list(set(ys_sorted_modified))

sorted_rects = []


lines = []

for i in ys_sorted_modified:
    line = []
    for j in rects:
        if(j[1]<=i+50 and j[1]>=i-50):
            line.append(j)
    lines.append(line)

def sort_line(line):
    xs = []
    for i in line:
        xs.append(i[0])
    xs.sort()
    sorted_line = []
    for i in xs:
        for j in line:
            if(j[0]==i):
                sorted_line.append(j)
    return sorted_line

page = []
for i in lines:
    page.append(sort_line(i))

final_rects = []
for i in page:
    for j in i:
        final_rects.append(j)

nums = []
for rect in final_rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    nums.append(nbr[0])
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)


print(nums)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
