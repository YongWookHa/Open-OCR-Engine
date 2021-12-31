import torch
import argparse
import time
import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, abort

from utils.utils import decode_image_from_string, sort_wordBoxes, get_vocab, load_setting
from models.craft import CRAFT
from models.deepTextRecog import DeepTextRecog
from datasets.deepTextRecog_dataset import AlignCollate


app = Flask(__name__)
__VALID_KEY__ = "OPEN-OCR-KEY"

def get_ocr_result(image: np.ndarray):
    """
    image: [H, W, C]
    """
    
    boxes = craft.predict(image)
    
    if not boxes: return '', []

    boxes = sort_wordBoxes(boxes)
    
    batch = []
    for i, box in enumerate(boxes):
        lx, ly, rx, ry = box
        cropped = image[ly:ry, lx:rx, :]
        cropped = Image.fromarray(cropped)
        cropped = cropped.convert('L' if cfg.input_channel==1 else 'RGB')
        cropped.save('test/{}.png'.format(i))
        batch.append([cropped, ''])
        
    collate = AlignCollate(cfg)
    batch, _ = collate(batch)
    text_for_pred = torch.LongTensor(batch.size(0), cfg.max_seq_len + 1)
    output = deepTextRecog(batch, text_for_pred, is_train=False)
    texts = deepTextRecog.get_text(output)
    
    return texts, boxes

def auth(func):
    def wrapper(*args, **kwargs):
        if request.method == 'POST':
            req = request.get_json(force=True)
            if 'key' in req and req['key'] == __VALID_KEY__:
                return func(*args, **kwargs)
            else:
                abort(code=401, descroption="Not Authorized")
    return wrapper

@auth
@app.route('/ocr', methods=['POST'])
def return_ocr_result():
    if request.method == 'POST':
        req = request.get_json(force=True)
        image = decode_image_from_string(req['image'])
        t = time.time()
        texts, coords = get_ocr_result(image)
        result = {
            'text' : texts,
            'time' : time.time()-t,
            'coord' : coords
        }
        return jsonify(result)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='host address')
    parser.add_argument('--port', type=str, default='5000', help='port number')
    parser.add_argument('--detector_ckpt', type=str, required=True, help='checkpoint path of detector')
    parser.add_argument('--recognizer_ckpt', type=str, required=True, help='checkpoint path of recognizer')
    parser.add_argument('--setting', type=str, required=True, help='checkpoint path of recognizer')
    parser.add_argument('--vocab', type=str, default='vocab.txt', help='vocab path')
    args = parser.parse_args()
    cfg = load_setting(args.setting)
    
    print('[LOADING] Detector from {}'.format(args.detector_ckpt))
    craft = CRAFT(cfg)
    craft = craft.load_from_checkpoint(args.detector_ckpt, cfg=cfg, strict=False)
    craft.freeze()
    
    print('[LOADING] Vocab from {}'.format(args.vocab))
    tokens = get_vocab(args.vocab)
    print('[LOADING] Recognizer from {}'.format(args.recognizer_ckpt))
    deepTextRecog = DeepTextRecog(cfg, tokens)
    deepTextRecog = deepTextRecog.load_from_checkpoint(args.recognizer_ckpt, cfg=cfg, tokens=tokens, strict=False)
    deepTextRecog.freeze()
    
    app.run(host=args.host, port=args.port)