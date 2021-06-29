import os
import numpy as np


def main(src_list, dst_list, new_prefix="Dev_images/"):
    """
    add prefix to each item
    """
    with open(src_list, 'r') as fr, open(dst_list, 'w') as fw:
        lines = fr.readlines()
        for i,line in enumerate(lines):

            fw.writelines(new_prefix + line)
            if (i % 1000 == 0):
                print("count:  ", i)
    return


if __name__ == "__main__":
    src_list = "/home/data4/OULU/list/dev_list.txt"
    dst_list = "/home/projects/face_liveness/FAS/data/list_oulu/dev_list.txt"
    main(src_list, dst_list)