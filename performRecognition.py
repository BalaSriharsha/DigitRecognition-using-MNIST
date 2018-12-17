import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

clf = joblib.load("digits_cls.pkl")

im = cv2.imread("photo_4.jpg")

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]

cor_sum = []
for i in rects:
    cor_sum.append(i[0]+i[1])
cor_dic = dict(zip(cor_sum,rects))
vals = []
for i,j in cor_dic.items():
    vals.append(i)
vals.sort()
vds = []
for i in vals:
    vds.append(cor_dic[i])

q = vds[0][1]

vs = []
for i in vds:
    if i[1]<=q+50 and i[1]>=q-50:
        vs.append(i)

nums = []

for rect in vs:
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

#print(vals)
print(nums)
#print(vds)
#print(cor_dic)
#print(q)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
