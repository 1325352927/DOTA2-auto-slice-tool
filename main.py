from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import speedx
from moviepy.video.fx.all import crop
from moviepy.editor import VideoFileClip
import cv2
import os
import numpy as np
import torch
from train.train import *
from tqdm import tqdm
import subprocess
import shutil
from datetime import datetime

def clear_folder():
    all_path = ["./temp/video_to_frame/", "./temp/cut_pic_get_last_dataset/", "./temp/get_dataset_num/", 
                "./temp/out/"]
    for folder_path in all_path:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # 清空文件夹内容
    print("清空中介文件夹中！")
    for i in tqdm(range(len(all_path))):
        folder_path = all_path[i]
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

def video_to_frame(video_path, identify_pic_gap, fps = 30):
    file = open("frame_name.txt", "w")
    file.write('')
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("提取关键帧图片中！")
    frame_num = 0
    for i in tqdm(range(frame_count)):
        ret = video.grab()
        if i % (identify_pic_gap * fps) != 0:
            continue
        if i % (identify_pic_gap * fps) == 0:
            ret, frame = video.retrieve()
            cv2.imwrite("./temp/video_to_frame/frame_{}.jpg".format((i) / (identify_pic_gap * fps)), frame)
            file.write("./temp/cut_pic_get_last_dataset/frame_{}.jpg".format(frame_num) + "\n")
            frame_num += 1
    video.release()
    file.close()


# 在图片中切割指定部分（数字）
def get_dataset_num():
    x, y, w, h = 858, 7, 55 , 25 # 左侧
    # x1, y1, w1, h1 = 1005, 7, 55, 25  # 右侧
    imagelist = os.listdir("./temp/video_to_frame/")
    i = 0
    print("获取一方比分区域图片中！")
    with tqdm(total=len(imagelist)) as pbar:
        for imgname in imagelist:
            if (imgname.endswith(".jpg")):
                image = cv2.imread("./temp/video_to_frame/"+imgname)
                height, width, _ = image.shape
                crop_img = image[y:y+h, x:x+w]
                cv2.imwrite('./temp/get_dataset_num/'+imgname, crop_img)
                i = i + 1
                pbar.update(1)

# 清除不是数字的图片，用于制作数据集
def clear_dataset_first(dataset_path):
    imagelist = os.listdir(dataset_path)
    a = 0
    for imgname in imagelist:
        a = 0
        image = cv2.imread(dataset_path+imgname)
        for i in range(25):
            for j in range(55):
                # if image[i, j][0] > 200 and image[i, j][1] > 150 and image[i, j][2] > 150:
                #     a = 1
                #     break
                if image[i, j][0] < 150:
                    a = 1
                    break
            if a == 1:
                break
        if a == 0:
            os.remove(dataset_path+imgname)
            print(imgname + "已删除")

# 得到二值化数字
def cut_pic_get_last_dataset():
    imagelist = os.listdir("./temp/get_dataset_num/")
    all_count = 0
    print("图片预处理中！")
    with tqdm(total=len(imagelist)) as pbar:
        for imgname in imagelist:
            image = cv2.imread("./temp/get_dataset_num/"+imgname)
            gray_img = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            thresh, new_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
            # 切割
            isGetWhite = 0
            left_point = 0
            last_point = 0
            is_cut = False
            for j in range(55):
                # blank_count = 0
                for i in range(25):
                    if (new_img[i, j] == 255 and isGetWhite == 0):
                        isGetWhite = 1
                        left_point = j
                        break
                    if (isGetWhite == 1):
                        if (new_img[i, j] == 255):
                            break
                        if (i == 24):
                            isGetWhite = 0
                            last_point = j
                            if last_point >= 50 or left_point < 1:
                                is_cut = False
                                break
                            else:   
                                is_cut = True
                            crop_img = new_img[0:25, left_point - 1:last_point +1]
                            cv2.imwrite("./temp/cut_pic_get_last_dataset/" +imgname, crop_img)
                            all_count += 1
                            pbar.update(1)
                            break 

            if (is_cut == False):
                cv2.imwrite("./temp/cut_pic_get_last_dataset/" +imgname, new_img)
                pbar.update(1)

def reshape_pic():
    imagelist = os.listdir("./temp/cut_pic_get_last_dataset/")
    print("图片归一化中！")
    with tqdm(total=len(imagelist)) as pbar:
        for imgname in imagelist:
            image = cv2.imread("./temp/cut_pic_get_last_dataset/"+imgname)
            image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_CUBIC)
            # os.remove(dataset_path+imgname)
            cv2.imwrite("./temp/cut_pic_get_last_dataset/"+imgname, image)
            pbar.update(1)

def identify_num_and_cut_video(video_path, identify_pic_gap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("./model/model.pth")
    model.eval()
    print("加载模型成功！")
    file = open("frame_name.txt", "r")
    i  = 1
    is_start = False
    is_identify_zero = False
    point = []
    tmp = 0
    with file as f:
        for line in f:
            one_content = line.strip()
            one_num = get_one_num(model, one_content, device)
            
            if is_identify_zero == False and one_num == 0:
                is_identify_zero = True

            if one_num >= 1 and is_start == False and one_num != 10 and is_identify_zero:
                is_start = True
                tmp = one_num
                point.append(i)
                
            if one_num != 10 and is_start and (one_num - tmp) != 0 and ((one_num - tmp) > 0 or (one_num - tmp) <= -6) :
                point.append(i)
                tmp = one_num
            i += 1
    file.close()
    print(point)
    all_count = len(point)
    cut_video_num = 0
    i = 0
    clip = VideoFileClip(video_path) 
    is_over = False
    file_video_name = open("frame_video_name.txt", "w")
    file_video_name.write('')
    print("获得切片中！")
    
    with tqdm(total=all_count) as pbar:
        while True:
            start_time = point[i] - 2 - 1
            end_time = point[i] - 1
            while True:
                i += 1
                if (i == all_count - 1):
                    is_over = True
                    break
                pbar.update(1)
                if (point[i] - point[i - 1] <= 2):
                    end_time = point[i] - 1
                else:
                    # 利用ffmepg，速度大概快60倍但有问题
                    # new_start_time = frame_to_time(start_time * identify_pic_gap + 5) 
                    # new_end_time = frame_to_time(end_time * identify_pic_gap + 2)
                    # output_video_name = "./temp/out/{}.mp4".format(cut_video_num)
                    # command = f"ffmpeg -i {video_path} -ss {new_start_time} -to {new_end_time} -c:v libx264  -avoid_negative_ts 1 {output_video_name}"
                    # subprocess.call(command, shell=True)
                    subclip = clip.subclip(start_time * identify_pic_gap + 5, end_time * identify_pic_gap + 2)
                    subclip.write_videofile("./temp/out/{}.mp4".format(cut_video_num))
                    file_video_name.write("./temp/out/{}.mp4".format(cut_video_num) + "\n")
                    cut_video_num += 1
                    break
            if is_over:
                break

    file_video_name.close()
        
def frame_to_time(frame):
    a = int(frame / 3600)
    frame = frame % 3600
    b = int(frame / 60)
    c = frame % 60
    output_time = "{:02d}:{:02d}:{:02d}".format(a, b, c)
    print(output_time)
    return output_time

def merge_video(output_video_name):
    # 指定要合并的视频文件列表
    with open("frame_video_name.txt", "r") as file:
        video_files = file.read().splitlines()

    # 指定合并后的视频文件
    output_video = "./temp/output_video/" + output_video_name + ".mp4"
    output_audio = "./temp/output_video/" + output_video_name + ".mp3"

    # 创建FFmpeg导出音频命令
    audio_command = f"ffmpeg -i {video_files[0]}"

    for i in range(1, len(video_files)):
        audio_command += f" -i {video_files[i]}"

    audio_command += f" -filter_complex \"[{':a'}]concat=n={len(video_files)}:v=0:a=1[outa]\" -map [outa] -c:a libmp3lame -qscale:a 2 {output_audio}"
    subprocess.call(audio_command, shell=True)

    # 创建FFmpeg合并视频和音频的命令
    command = f"ffmpeg -i {video_files[0]}"

    for i in range(1, len(video_files)):
        command += f" -i {video_files[i]}"

    command += f" -i {output_audio} -filter_complex \"[{':v' * len(video_files)}][{':a'}]concat=n={len(video_files)}:v=1:a=1[outv]\" -map [outv] -c:v libx264 -crf 23 -preset veryfast -c:a aac -b:a 128k {output_video}"

    # 调用FFmpeg命令
    subprocess.call(command, shell=True)
    

# 先宽度后长度
if __name__ == "__main__":
    video_path = "./video/2.mp4"  # 待处理视频路径
    output_video_name = "test"
    identify_pic_gap = 10   # 处理间隔；例如 10：表示每10s取一张图识别，数字越小，捕捉团战越精确，同时处理时间越长
    fps = 30
    # 开始处理
    clear_folder()
    video_to_frame(video_path, identify_pic_gap)
    get_dataset_num()
    cut_pic_get_last_dataset()
    reshape_pic()
    identify_num_and_cut_video(video_path, identify_pic_gap)
    merge_video(output_video_name)
    
    

