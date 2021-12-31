import random as rnd
import numpy as np
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter

def generate(text, font, text_color, font_size, orientation, space_width, fit):
    if orientation == 0:
        # only horizontal_text supports bounding box
        return _generate_horizontal_text(text, font, text_color, font_size, space_width, fit)
    elif orientation == 1:
        return _generate_vertical_text(text, font, text_color, font_size, space_width, fit)
    else:
        raise ValueError("Unknown orientation " + str(orientation))

def _generate_horizontal_text(text, font, text_color, font_size, space_width, fit):
    image_font = ImageFont.truetype(font=font, size=font_size)
    words = text.split(' ')
    space_width = image_font.getsize(' ')[0] * space_width 

    words_width = [image_font.getsize(w)[0] for w in words]
    text_width =  sum(words_width) + int(space_width) * (len(words) - 1)
    text_height = max([image_font.getsize(w)[1] for w in words])
    
    char_bbox = []
    ac = 0 # accumulator

    for word in words:
        for char in word:
            w, h = image_font.getsize(char)  # width, height
            # dx, dy = image_font.getoffset(char)
            # bbox = [[ac+dx, dy], [ac+w, dy], [ac+w, h], [ac+dx, h]]
            bbox = [[ac, 0], [ac+w, 0], [ac+w, h], [ac, h]]
            
            char_bbox.append(bbox)
            ac += image_font.getsize(char)[0] 
        ac += int(space_width)

    txt_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))

    txt_draw = ImageDraw.Draw(txt_img)

    colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2]))
    )

    for i, w in enumerate(words):
        txt_draw.text((sum(words_width[0:i]) + i * int(space_width), 0), w, fill=fill, font=image_font)
        
    if fit:
        return txt_img.crop(txt_img.getbbox()), char_bbox
    else:
        return txt_img, char_bbox

def _generate_vertical_text(text, font, text_color, font_size, space_width, fit):
    image_font = ImageFont.truetype(font=font, size=font_size)
    
    space_height = int(image_font.getsize(' ')[1] * space_width)

    char_heights = [image_font.getsize(c)[1] if c != ' ' else space_height for c in text]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights)

    char_bbox = []
    ac = 0 # accumulator
    words = text.split(' ')
    for word in words:
        for char in word:
            w, h = image_font.getsize(char)  # width, height
            # dx, dy = image_font.getoffset(char)
            # bbox = [[dx, ac+dy], [w, ac+dy], [w, ac+h], [dx, ac+h]]
            bbox = [[0, ac], [w, ac], [w, ac+h], [0, ac+h]]
            char_bbox.append(bbox)
            ac += image_font.getsize(char)[1] 
        ac += int(space_height)

    txt_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))

    txt_draw = ImageDraw.Draw(txt_img)

    colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2])
    )

    with open('font_log.txt', 'a', encoding='utf8') as f:
        f.write(text+'\t'+font+'\n')

    for i, c in enumerate(text):
        txt_draw.text((0, sum(char_heights[0:i])), c, fill=fill, font=image_font)

    if fit:
        return txt_img.crop(txt_img.getbbox()), char_bbox
    else:
        return txt_img, char_bbox
