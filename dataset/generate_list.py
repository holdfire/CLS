import os

def generate_list_of_images(dir_path, dst_file):
    """
    ############# depth first search
    dir_path = "/home/data2/CASIA-HiFi-Mask-crop"
    dst_file =  "/home/projects/list/CASIA-HiFi-Mask-crop.txt"
    generate_list_of_images(dir_path, dst_file)
    """
    count = 0
    with open(dst_file, "w") as fw:
        res = []
        dfs(dir_path, res)
        for line in res:
            fw.writelines(line + "\n")
            count += 1
            if count % 100 == 0:
                print("current count: ", count)
    return 

def dfs(path, res):
    # dst_dict = ["bmp", "jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]
    dst_dict = ["avi", "mp4"]
    if os.path.isfile(path) and path.split(".")[-1] in dst_dict:
        res.append(path)
        # print(len(res))
    elif os.path.isdir(path):
        for dirname in os.listdir(path):
            path2 = os.path.join(path, dirname)
            dfs(path2, res)


if __name__ == '__main__':
    
    dir_path = "/home/data/usb/face_antispoofing_data/shu_ju_part/"
    dst_file = "/home/data/usb/face_antispoofing_data/count.txt"
    generate_list_of_images(dir_path, dst_file)
