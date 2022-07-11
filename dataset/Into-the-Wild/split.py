from moviepy.editor import VideoFileClip
from math import ceil
from glob import glob
from tqdm import tqdm
import os

def video_clip_segmentation(file_path, save_path):
    clip = VideoFileClip(file_path)
    duration = clip.duration
    iteration = ceil(duration / 3)
    for i in range(iteration):
    	subclip = clip.subclip(i*3,(i+1)*3)
    	subclip.write_videofile(save_path+file_path.split('/')[-1].split('.')[0]+'_clip{}'.format(i+1)+'.mp4')

if __name__ == '__main__':
    video_path = './youtube'
    video_chunk_path = './youtube_chunk'
    if not os.path.exists(video_chunk_path):
        os.makedirs(video_chunk_path)

    video_list = sorted(glob(video_path))
    for video in tqdm(video_list):
        video_clip_segmentation(file_path=video, save_path=video_chunk_path)