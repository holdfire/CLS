import os
import numpy as np
import cv2

def main(src_dir, dst_list, save_dir, save_count=100):
    """
    add prefix to each item
    """
    with open(dst_list, 'r') as fr:
        lines = fr.readlines()
        for i,line in enumerate(lines):
            name = line.split(" ")[0]
            bbox = line.strip().split(" ")[2:]
            bbox = [int(i) for i in bbox]
            if i < save_count:
                img_path = os.path.join(src_dir, name)
                img = cv2.imread(img_path)
                pt1 = (bbox[0], bbox[1])
                pt2 = (bbox[0]+bbox[2], bbox[1]+bbox[3])
                img = cv2.rectangle(img, pt1, pt2, (0,255,0), 1)
                # print(pt1)
                # print(pt2)
                # cv2.rectangle(img, (330, 80), (450, 220), (0, 255, 0), 2)
                save_path = os.path.join(save_dir, name)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                cv2.imwrite(save_path, img)
                if (i % 10 == 0):
                    print("count:  ", i)
    return





if __name__ == "__main__":
    src_dir = "/home/data/data/HiFiMask-Challenge/"
    dst_list = "/home/data4/OULU/list/train_list.txt"
    save_dir = "/home/data4/OULU/Train_bbox/"
    main(src_dir, dst_list, save_dir)
