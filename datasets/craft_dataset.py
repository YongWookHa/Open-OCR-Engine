import torch
import cv2
import json
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms

from utils.craft_utils import loadImage
from utils.data_manipulation import generate_affinity, generate_target


class DatasetSYNTH(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(DatasetSYNTH, self).__init__()

        self.cfg = cfg
        self.dataPath = Path(cfg.data_path)
        self.basePath = self.dataPath.parent

        with self.dataPath.open('rb') as f:
            dsets = pickle.load(f)

        self.imnames, self.charBB, self.txt = [], [], []
        for d in tqdm(dsets, total=len(dsets), desc="loading dataset"):
            self.imnames.append(d['fn'])
            self.charBB.append(d['charBB'])
            self.txt.append(d['txt'])

    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, idx):
        image_fn = self.basePath / self.imnames[idx]
        image = loadImage(image_fn)
        char_boxes = np.array(self.charBB[idx]).astype(int)
        words = self.txt[idx]

        return image, char_boxes, words, image_fn


class CustomCollate(object):
    def __init__(self, image_size, load_preprocessed_data=False, save_data=False):
        self.image_size = image_size
        self.load_preprocessed_data = load_preprocessed_data
        self.save_data = save_data

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, batch):
        """
        preprocess batch
        """
        batch_big_image, batch_weight_character, batch_weight_affinity = [], [], []

        if self.load_preprocessed_data:
            for _, _, _, fn in batch:
                big_image = np.load(fn.parent / f"{fn.stem}_{self.image_size}_image.npy")
                weight_character = np.load(fn.parent / f"{fn.stem}_{self.image_size}_weight_character.npy")
                weight_affinity = np.load(fn.parent / f"{fn.stem}_{self.image_size}_weight_affinity.npy")

                batch_big_image.append(self.image_transform(Image.fromarray(big_image)))
                batch_weight_character.append(weight_character)
                batch_weight_affinity.append(weight_affinity)
        else:
            for image, char_boxes, words, fn in batch:
                big_image, small_image, character = self.resize(image, char_boxes, big_side=self.image_size)  # Resize the image

                # Generate character heatmap
                weight_character = generate_target(small_image.shape, character.copy())

                # Generate affinity heatmap
                weight_affinity, _ = generate_affinity(small_image.shape, character.copy(), words)

                weight_character = weight_character.astype(np.float32)
                weight_affinity = weight_affinity.astype(np.float32)

                if self.save_data:
                    np.save(fn.parent / f"{fn.stem}_{self.image_size}_image.npy" , big_image)
                    np.save(fn.parent / f"{fn.stem}_{self.image_size}_weight_character.npy", weight_character)
                    np.save(fn.parent / f"{fn.stem}_{self.image_size}_weight_affinity.npy", weight_affinity)

                batch_big_image.append(self.image_transform(Image.fromarray(big_image)))
                batch_weight_character.append(weight_character)
                batch_weight_affinity.append(weight_affinity)

        return  torch.stack(batch_big_image),  \
                torch.from_numpy(np.stack(batch_weight_character)),  \
                torch.from_numpy(np.stack(batch_weight_affinity))

    def resize(self, image, character, big_side):
        """
            Resizing the image while maintaining the aspect ratio and padding with average of the entire image to make the
            reshaped size = (side, side)
            :param image: np.array, dtype=np.uint8, shape=[height, width, 3]
            :param character: np.array, dtype=np.int32 or np.float32, shape = [2, 4, num_characters]
            :param side: new size to be reshaped to
            :return: resized_image, corresponding reshaped character bbox
        """

        height, width, channel = image.shape
        max_side = max(height, width)
        big_resize = (int(width/max_side*big_side), int(height/max_side*big_side))
        small_resize = (int(width/max_side*(big_side//2)), int(height/max_side*(big_side//2)))
        image = cv2.resize(image, big_resize)

        character = np.array(character)
        character[0, :, :] = character[0, :, :] * (small_resize[0] / width)
        character[1, :, :] = character[1, :, :] * (small_resize[1] / height)

        big_image = np.ones([big_side, big_side, 3], dtype=np.float32)*255
        h_pad, w_pad = (big_side-image.shape[0])//2, (big_side-image.shape[1])//2
        big_image[h_pad: h_pad + image.shape[0], w_pad: w_pad + image.shape[1]] = image
        big_image = big_image.astype(np.uint8)

        small_image = cv2.resize(big_image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        character[0, :, :] += (w_pad // 2)
        character[1, :, :] += (h_pad // 2)

        # character fit to small image
        return big_image, small_image, character
