import numpy as np
from PIL import Image
import os
import math
# import cv2

seg_dir = ''
label_file = ''
lines = [line.rstrip() for line in open(label_file, 'r')]

result_file = ''
result = open(result_file,'w')

for img in os.listdir(seg_dir):
    img_path = os.path.join(seg_dir, img)
    image = Image.open(img_path)
    img_np = np.array(image)[:,:,0]
    idx = np.argmax(img_np)
    y_coord = idx / 490
    x_coord = idx - y_coord * 490
    result_x = x_coord / float(490)
    result_y = y_coord / float(326)

    for i, line in enumerate(lines):
        if line.split(' ')[0] == img:
            gt_x = float(line.split(' ')[1])
            gt_y = float(line.split(' ')[2])
            x_err = abs(result_x - gt_x)
            y_err = abs(result_y - gt_y)
            dist_err = math.sqrt(x_err**2 + y_err**2)
            print('image name: {}, pred x: {}, pred y: {}, gt x: {}, gt y:{}, dist_err: {}'.format(
                img, result_x, result_y, gt_x, gt_y, dist_err
            ))
            result.write(img + ' ' + str(x_coord) + ' ' + str(y_coord) + ' ' + str(dist_err) + '\n')

result.close()
