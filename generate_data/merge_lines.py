import  numpy as np
import sys
import random
import pickle
import argparse
from tqdm import tqdm

from PIL import Image, ImageDraw
from pathlib import Path


def cutList(li, mini=2, maxi=6):
    chunk_size = []
    while sum(chunk_size) <= len(li):
        if len(li)-sum(chunk_size) <= maxi:
            chunk_size.append(len(li)-sum(chunk_size))
            break
        cs = random.randint(mini, maxi)
        chunk_size.append(cs)

    res = []
    acc = 0
    for cs in chunk_size:
        res.append(li[acc:acc+cs])
        acc += cs

    return res

def combineHorizontal(chunk_list:"list of img data", base, height):
    res = []
    for idx, chunk in tqdm(enumerate(chunk_list), total=len(chunk_list)):
        imgs = [Image.open(Path(base) / data['fn']) for data in chunk]
        charBB = [data['charBB'] for data in chunk]
        txt = [data['txt'] for data in chunk]

        resize_ratio = [height/img.size[1] for img in imgs]
        resized_imgs = [img.resize(tuple(map(lambda x: round(x*resize_ratio[i]), img.size))) for i, img in enumerate(imgs)]
        resized_imgs = [np.asarray(img) for img in resized_imgs]

        resized_charBB = [np.multiply(bb, resize_ratio[i]) for i, bb in enumerate(charBB)]
        accumulated_charBB, acc = [], 0
        for i, bb in enumerate(resized_charBB):
            foo = np.dstack((np.add(bb[0],acc), bb[1]))
            bar = np.rollaxis(foo, axis=2)
            acc += resized_imgs[i].shape[1]
            accumulated_charBB.append(bar)

        combined_img = Image.fromarray(np.hstack(resized_imgs))
        combined_charBB = np.dstack(accumulated_charBB)
        combined_txt = "\n".join(txt)

        combined_img_path = Path(base) / 'combined' / (str(idx)+'.jpg')

        # pdraw = ImageDraw.Draw(combined_img)
        # for bbox in np.transpose(combined_charBB, (2,1,0)):
        #     foo = [tuple(pt) for pt in bbox]
        #     pdraw.line(foo+foo[:1], fill='red', width=2)

        combined_img.save(str(combined_img_path))

        new_data = {
            "fn" : str(combined_img_path.name),
            "charBB" : combined_charBB,
            "txt" : combined_txt
        }
        res.append(new_data)
    return res

def combineVertical(chunk_list:"list of img data", base, width):
    res = []
    for idx, chunk in tqdm(enumerate(chunk_list), total=len(chunk_list)):
        imgs = [Image.open(Path(base) / data['fn']) for data in chunk]
        charBB = [data['charBB'] for data in chunk]
        txt = [data['txt'] for data in chunk]

        resize_ratio = [width/img.size[0] for img in imgs]
        resized_imgs = [img.resize(tuple(map(lambda x: round(x*resize_ratio[i]), img.size))) for i, img in enumerate(imgs)]
        resized_imgs = [np.asarray(img) for img in resized_imgs]

        resized_charBB = [np.multiply(bb, resize_ratio[i]) for i, bb in enumerate(charBB)]
        accumulated_charBB, acc = [], 0
        for i, bb in enumerate(resized_charBB):
            foo = np.dstack((bb[0],np.add(bb[1],acc)))
            bar = np.rollaxis(foo, axis=2)
            acc += resized_imgs[i].shape[0]
            accumulated_charBB.append(bar)

        combined_img = Image.fromarray(np.vstack(resized_imgs))
        combined_charBB = np.dstack(accumulated_charBB)
        combined_txt = "\n".join(txt)

        combined_img_path = Path(base) / 'combined' / (str(idx)+'.jpg')

        # pdraw = ImageDraw.Draw(combined_img)
        # for bbox in np.transpose(combined_charBB, (2,1,0)):
        #     foo = [tuple(pt) for pt in bbox]
        #     pdraw.line(foo+foo[:1], fill='red', width=2)

        combined_img.save(str(combined_img_path))

        new_data = {
            "fn" : str(combined_img_path.name),
            "charBB" : combined_charBB,
            "txt" : combined_txt
        }
        res.append(new_data)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--orientation', type=str, choices=['horizontal', 'vertical'],
                        required=True, help='orientation to merge')
    parser.add_argument('-b', '--base', type=str, required=True, help='base path of images')
    parser.add_argument('--width', type=int, help='fixed width when vertical merge')
    parser.add_argument('--height', type=int, help='fixed height when horizontal merge')
    parser.add_argument('--min', type=int, required=True, help='min number of images to merge')
    parser.add_argument('--max', type=int, required=True, help='max number of images to merge')

    args = parser.parse_args()

    imgs = list(Path(args.base).glob('*.jpg'))

    print('imgs : ', len(imgs))

    total_data = []

    for img in imgs:
        pkl = img.with_suffix('.pkl')
        with pkl.open('rb') as f:
            data = pickle.load(f)

        this_data = {
            'fn' : img.name,
            'charBB' : data['charBB'],
            'txt' : data['txt']
        }
        total_data.append(this_data)

    (Path(args.base) / 'combined').mkdir(parents=True, exist_ok=True)

    chunk = cutList(total_data, mini=args.min, maxi=args.max)
    if args.orientation == 'vertical':
        combined = combineVertical(chunk, args.base, args.width)
    elif args.orientation == 'horizontal':
        combined = combineHorizontal(chunk, args.base, args.height)


    with (Path(args.base)/'combined'/'merged_gt.pkl').open('wb') as gt:
        pickle.dump(combined, gt)

