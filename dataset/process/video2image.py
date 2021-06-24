"""
video processing utils, created by Liu Xing on 2020/05/20. including:
video2images
"""
import os
import cv2

def video2images(video_path, image_save_dir, frame_interval=3, resize=False, new_size=(128, 128)):
    """
    把视频按一定的帧间隔保存为图片
    """
    video = cv2.VideoCapture(video_path)
    count = 0
    rval = video.isOpened()
    while rval:
        count = count + 1
        rval, frame = video.read()
        if count % frame_interval != 0:
            continue
        if resize:
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
        # 最后一帧为空，需要丢弃掉
        if frame is not None:
            image_name = video_path + "_" + str(count) + ".png"
            image_path = os.path.join(image_save_dir, image_name)
            cv2.imwrite(image_path, frame)
    return image_save_dir



def video2images2(video_path, save_dir, frames=10):
    """
    把视频按保存为固定帧数的图片
    """
    # 确定单个视频解析保存目录，如果已经有10张了，就直接跳过
    video_name = os.path.basename(video_path).split(".")[0]
    img_save_dir = os.path.join(save_dir, video_name)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if os.path.exists(img_save_dir) and len(os.listdir(img_save_dir)) == frames:
        print("already exist!")
        return None

    # 首先确定该视频一共有多少帧，以确定隔多少帧取
    video = cv2.VideoCapture(video_path)
    total = 0
    rval = video.isOpened()
    while rval:
        total += 1
        rval, frame = video.read()
        if frame is None:
            total -= 1
    frame_interval = total // frames    # 确定取帧的间隔数

    # 然后重新读取图片，这个做法有点蠢啊！！！目前还没想到好的解决算法
    video2 = cv2.VideoCapture(video_path)
    count = 0                           # 当前帧的index
    saved_count = 0                     # 要保存帧的index
    rval = video2.isOpened()
    while rval:
        count = count + 1
        rval, frame = video2.read()
        if count % frame_interval != 0:
            continue
        if saved_count > frames:
            break
      
        if frame is not None:            # 最后一帧为空，需要丢弃掉
            image_name = str(saved_count) + ".png"
            image_path = os.path.join(img_save_dir, image_name)
            cv2.imwrite(image_path, frame)
            saved_count += 1
    
    # print("complete {} ,name: {}".format(str(saved_count), image_path))
    return img_save_dir



def batch_video2images2(video_list, save_dir):
    """
    批量处理视频
    """
    with open(video_list, 'r') as fr:
        lines = fr.readlines()
        for i,line in enumerate(lines):
            video_path = line.strip()
            video2images2(video_path, save_dir)
            print("processing: {} / {}".format(str(i), str(len(lines))))
    return




if __name__ == "__main__":

    # video_path = "/home/data/data/oulu/Train_files/6_1_14_5.avi"
    # save_dir = "/home/data/data/oulu/Train_images/"
    # video2images2(video_path, save_dir)

    video_list = "/home/data/data/oulu/list/dev_videos.txt"
    save_dir = "/home/data/data/oulu/Dev_images/"
    batch_video2images2(video_list, save_dir)

