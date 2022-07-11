import os
import torchvision
import math
import torch.nn.functional as F
import numpy as np
import tqdm
import math
import torch
import soundfile as sf
from  torchaudio.transforms import Resample
import argparse
from imageio import imread, imsave
from glob import glob

def video2jpg(video_path, jpg_path, audio_path, start, end, sr=16000, num_frames=8):
	video_list = sorted(os.listdir(video_path))[start:end]
	for i in tqdm.tqdm(video_list):
		v, a, info = torchvision.io.read_video(os.path.join(video_path, i), pts_unit='sec')
		try:
			v = v.repeat(math.ceil(num_frames/v.shape[0]),1,1,1)
		except:
			print(os.path.join(video_path, i))
			continue
		total_frames = v.shape[0]
		v = v[::total_frames//num_frames,:,:,:][:num_frames,:,:,:]
		v = F.interpolate(v.float().permute(0, 3, 1, 2), (512, 512), mode='bilinear', align_corners=False)  # [T, C, H, W]
		v = v.permute(0,2,3,1) # [T, H, W, C]
		images = v.numpy().astype(np.uint8)
		resample = Resample(orig_freq=info["audio_fps"], new_freq=sr)
		a = resample(a).squeeze(0)
		if a.shape[0] == 2:
			a = torch.mean(a, dim=0).numpy()
		elif a.shape[0] == 1:
			a = a.squeeze(0).numpy()
		else:
			a = a.numpy()
		for num, image in enumerate(images):
			imsave(os.path.join(jpg_path, i[:-4]+'-'+str(num+1)+'.jpg'), image)
		sf.write(os.path.join(audio_path, i[:-3]+'wav'), a, samplerate=sr)
		with open('./audio_list.txt','a+') as f:
			f.write(os.path.join(audio_path, i[:-3]+'wav') + "\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--s', type=int, default=0, help='start number')
	parser.add_argument('--e', type=int, default=1000000, help='end number')
	args = parser.parse_args()

	video_path = "./youtube_chunk/"
	jpg_path = "./youtube_chunk/jpg/"
	audio_path = "./audio/"
	if not os.path.exists(jpg_path):
		os.makedirs(jpg_path)
	if not os.path.exists(audio_path):
		os.makedirs(audio_path)
	video2jpg(video_path=video_path, jpg_path=jpg_path, audio_path=audio_path, start=args.s, end=args.e)
