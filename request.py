import base64
import requests
import sys
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path


__VALID_KEY__ = "OPEN-OCR-KEY"
API_HOST = 'http://127.0.0.1:5000/ocr'  # your OCR-Engine server url
target = Path(sys.argv[1])

def request_OCR_API(filename: Path.__class__):
    with filename.open('rb') as f:
        img = f.read()
        
    x = base64.encodebytes(img).decode('utf8')

    inp = {
        'key' : __VALID_KEY__,
        'id' : '0001',
        'data_type' : 'byte-string',
        'image' : "data:image/jpeg;base64," + x
    }
    
    res = requests.post(API_HOST, json=inp)
    
    if res:
        return res.json()
    else:
        raise ValueError
    
def drawSample(fn, coord):
    sample = Image.open(fn)
    draw = ImageDraw.Draw(sample)
    for x in coord:
        draw.rectangle(x, outline='blue', width=10)
    Path('detect_result').mkdir(exist_ok=True)
    sample.save(Path('detect_result') / fn.name)

if __name__ == "__main__":
    res = request_OCR_API(target)
    print(res)
    drawSample(target, res['coord'])