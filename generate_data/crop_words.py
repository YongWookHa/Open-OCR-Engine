import cv2
import numpy as np
import pickle
import argparse
from PIL import Image
from multiprocessing import Manager, cpu_count, Pool

from pathlib import Path

def crop_words(d, res, im_dir, out_dir):
    fn = d['fn']
    txt = d['txt'].strip().split()
    boxes = d['charBB']
    # image = cv2.imread(str(im_dir/fn))
    image = Image.open(im_dir/fn).convert('RGB')
    image = np.array(image)

    if image is None:
        raise ValueError
    cnt1, cnt2 = 0, 0
    for t in txt:
        if cnt1 >= boxes.shape[-1]:
            break

        lx = int(boxes[:,:,cnt1: cnt1+len(t)].transpose()[:,0,0].min())
        ly = int(boxes[:,:,cnt1: cnt1+len(t)].transpose()[:,0,1].min())

        rx = int(boxes[:,:,cnt1: cnt1+len(t)].transpose()[:,2,0].max())
        ry = int(boxes[:,:,cnt1: cnt1+len(t)].transpose()[:,2,1].max())

        # lx, ly = list(map(int, boxes[:,:,cnt1].transpose()[0]))
        # rx, ry = list(map(int, boxes[:,:,cnt1+length].transpose()[2]))
        cnt1 += len(t)

        cropped = image[ly:ry, lx:rx, :]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            break

        cv2.imwrite(str(out_dir / "{}_{}.png".format(fn[:-4], cnt2)), cropped)
        res.append(("{}_{}.png".format(fn[:-4], cnt2), t))
        cnt2 += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    im_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cpu = cpu_count()-2
    manager = Manager()
    res = manager.list()
    pool = Pool(processes=num_cpu)

    with Path(args.pickle).open('rb') as pkl:
        data = pickle.load(pkl)

    pool.starmap(crop_words, [(d, res, im_dir, out_dir) for d in data])
    pool.close()
    pool.join()

    print('len(res):', len(res))

    with (out_dir / 'gt.pkl').open('wb') as f:
        pickle.dump(list(res), f)
