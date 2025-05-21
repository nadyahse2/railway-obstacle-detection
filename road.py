import os
import cv2
import numpy as np
import math
import sys

def clear_directory():
    path_dir = 'results'

    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    for filename in os.listdir(path_dir):
            file_path = os.path.join(path_dir,filename)
            os.unlink(file_path)
def is_rectangle(contour):
    if len(contour) <= 7:
        return True
    approx = cv2.approxPolyDP(contour, 5, True)
    return len(approx) <= 7


def len_cal(con):
        x_coords = [pnt[0][0] for pnt in con]
        y_coords = [pnt[0][1] for pnt in con]

        x_max = max(x_coords)
        x_min = min(x_coords)
        y_max = max(y_coords)
        y_min = min(y_coords)

        return np.sqrt((y_max - y_min) ** 2 + (x_max - x_min) ** 2)

def write_photo(name,img):
    filepath = os.path.join('results',name)
    cv2.imwrite(filepath, img)

def distance_point_to_line(point, line):
    x0, y0 = point
    x1, y1, x2, y2 = line
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den if den != 0 else float('inf')

path = sys.argv[1]
img = cv2.imread(path)
clear_directory()
height, weight = img.shape[:2]
write_photo('original.jpg',img)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
write_photo('without_cleaning.jpg',th_img)


contours, _ = cv2.findContours(th_img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i in contours:
    s = cv2.contourArea(i)
    rect = cv2.minAreaRect(i)
    width2, height2 = rect[1]
    ratio = 0
    if height2 != 0 and width2 != 0:
        ratio = min(width2, height2) / max(width2, height2)
    len1 = cv2.arcLength(i, False)
    if s <= 100 or ratio >=0.1 or not is_rectangle(i):
        cv2.drawContours(th_img, [i], -1, 255, thickness=cv2.FILLED)


write_photo('clean.jpg',th_img)

inv_img = cv2.bitwise_not(th_img)
con, _ = cv2.findContours(inv_img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
con2,_ = cv2.findContours(th_img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(img_gray)[0]


min_length = min(height,weight)/3
filtered_lines = []
if lines is not None:
    for dline in lines:
        x1 = int(round(dline[0][0]))
        y1 = int(round(dline[0][1]))
        x2 = int(round(dline[0][2]))
        y2 = int(round(dline[0][3]))
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length >= min_length:
            filtered_lines.append((x1,y1,x2,y2))


matched_contours = set()
matched_contours2 = set()
matched_lines = []
if filtered_lines is not None:
    for line in filtered_lines:
        x1, y1, x2, y2 = line
        for i, c in enumerate(con):
            distances = [distance_point_to_line((pnt[0][0],pnt[0][1]), (x1, y1, x2, y2)) for pnt in c]
            middle_d = sum(distances)/len(distances)
            if middle_d < 10:
                matched_contours.add(i)
                matched_lines.append(line)
        for i, c in enumerate(con2):
            distances2 = [distance_point_to_line((pnt[0][0],pnt[0][1]), (x1, y1, x2, y2)) for pnt in c]
            middle_d2 = sum(distances2)/len(distances2)
            if middle_d2 < 10:
                matched_contours2.add(i)
                matched_lines.append(line)

rail_map = np.zeros(img.shape[:2], dtype=np.uint8)
output_image = img.copy()
for contour_idx in matched_contours:
    cv2.drawContours(rail_map, [con[contour_idx]], -1, 255, 2)
    cv2.drawContours(output_image, [con[contour_idx]], -1, (0, 0, 255), 2)
for contour_idx in matched_contours2:
    cv2.drawContours(rail_map, [con2[contour_idx]], -1, 255, 2)
    cv2.drawContours(output_image, [con2[contour_idx]], -1, (0, 0, 255), 2)
for line in matched_lines:
    x1, y1, x2, y2 = line
    cv2.line(rail_map, (x1, y1), (x2, y2), 255, 2)
    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

write_photo('l.jpg',output_image)

con3, _ = cv2.findContours(rail_map,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i in con3:
    if cv2.contourArea(i) < 100:
        cv2.drawContours(th_img, [i], -1, 0, thickness=cv2.FILLED)
kernel = np.ones((5,5), np.uint8)
dilated_map = cv2.dilate(rail_map, kernel, iterations=3)
closed_map = cv2.morphologyEx(dilated_map, cv2.MORPH_CLOSE, kernel, iterations=2)
write_photo('last.jpg',closed_map)
con_morf, _ = cv2.findContours(closed_map,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = 0
for i in con_morf:
    length_con = len_cal(i)
    if length_con > min(height,weight):
        count += 1
    elif abs(length_con-min(height,weight)) <50:
        count += 1
if count == 2 or count == 4:
    print('Завал нет')
else:
    print('Завал есть')

