import pickle
import math
import torch
import re

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

class DatasetSYNTH(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        with open(cfg.data_path, 'rb') as f:
            data = pickle.load(f)

        self.imagePath = Path(cfg.data_path).parent

        self.imnames, self.txt = [], []
        for fn, label in data:
            if len(label) <= cfg.dtr.max_seq_len:
                self.imnames.append(fn)
                self.txt.append(label)
            else:
                print(f'label of {fn} is longer than max_seq_len')

        self.tokens = set()
        for txt in self.txt:
            if not cfg.dtr.is_character:
                txt = txt.split(' ')
            for token in txt:
                self.tokens.add(token)
        self.tokens = list(self.tokens)

        with open('vocab.txt', 'w', encoding='utf8') as f:
            f.write(' '.join(self.tokens))

    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, item):
        item = item % len(self.imnames)
        image = Image.open(self.imagePath / self.imnames[item])  # Read the image
        image = image.convert('L' if self.cfg.dtr.input_channel==1 else 'RGB')
        txt = self.txt[item]

        return image, txt

class AlignCollate(object):
    def __init__(self, cfg):
        self.cfg = cfg.dtr
        self.imgH = self.cfg.imgH
        self.imgW = self.cfg.imgW
        self.keep_ratio_with_pad = self.cfg.PAD

    def __call__(self, batch):
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            transform = NormalizePAD((self.cfg.input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)  # -1.0 ~ 1.0
        return img

class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img
