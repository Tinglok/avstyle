import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import util.util as util
import librosa


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.A_paths = util.read_txt("datasets/Into-the-Wild/trainA.txt")
        self.B_paths = util.read_txt("datasets/Into-the-Wild/trainB.txt")
        self.wav_paths = os.path.join(opt.dataroot, "audio")
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def read_wav(self, wav_path, sr=16000, mono=True):
        try:
            y, sr = librosa.load(wav_path, sr=sr, mono=True)
        except:
            raise AssertionError("Unsupported file: %s" % wav_path)
        return y

    def wav_normalize(self, wav):
        norm = np.max(np.abs(wav)) * 1.1
        wav = wav/norm
        return wav

    def wav_tile(self, wav, out_shape=48000):
        repeat = int(out_shape / wav.shape[0])
        if repeat > 0:
            wav = np.tile(wav, (repeat + 1))
        return wav[:out_shape]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size].strip()  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B].strip()

        wav_path = os.path.join(self.wav_paths, B_path.split('/')[-1].split('clip')[0] + "clip" + B_path.split('/')[-1].split('clip')[1].split('-')[0] +".wav")
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        B_wav = self.read_wav(wav_path)
        B_wav = self.wav_normalize(self.wav_tile(B_wav))

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'B_wav': B_wav, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        # return min(self.A_size, self.B_size)
        return max(self.A_size, self.B_size)

