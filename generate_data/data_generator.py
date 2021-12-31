import random as rnd
import numpy as np
import pickle

from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path

import computer_text_generator
import background_generator
import distorsion_generator
try:
    import handwritten_text_generator
except ImportError as e:
    print('Missing modules for handwritten text generation.')

def rotate_bboxes(bboxes, degree, rc):
    def rotate_pts(pts, expand):
        nonlocal theta, rc
        rc, pts = np.array(rc), np.array(pts)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))
        res = (rc + R @ (pts - rc)).tolist()
        return [res[0], res[1]+expand]

    res = []
    theta = -degree * np.pi / 180
    if degree < 0:
        expand = -rotate_pts(bboxes[-1][2], 0)[1]+ bboxes[-1][2][1]  
    else:
        expand = rotate_pts((0,0), 0)[1]
    
    for bbox in bboxes:
        rc = ((bbox[2][0] - bbox[0][0])/2, (bbox[2][1] - bbox[0][1])/2)
        rotated_bbox = [rotate_pts(pts, expand) for pts in bbox]
        res.append(rotated_bbox)
    return res

def resize_bboxes(bboxes, ratio, vm, hm):  # vertical_margin, horizontal_margin
    res = []
    for bbox in bboxes:
        new_bbox = []
        for pts in bbox:
            new_pts = (int(pts[0]*ratio + hm/2), int(pts[1]*ratio + vm/2))
            new_bbox.append(new_pts)
        res.append(new_bbox)
    return res

class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(cls, index, text, font, out_dir, size, extension, skewing_angle, random_skew, blur, random_blur, background_type, distorsion_type, distorsion_orientation, is_handwritten, name_format, width, alignment, text_color, orientation, space_width, margins, fit, is_bbox, label_only):
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image = handwritten_text_generator.generate(text, text_color, fit)
        else:
            if orientation == 0:  
                image, bboxes = computer_text_generator.generate(text, font, text_color, size, orientation, space_width, fit)
            else:
                image, bboxes = computer_text_generator.generate(text, font, text_color, size, orientation, space_width, fit)

        random_angle = rnd.randint(0-skewing_angle, skewing_angle)

        rotated_img = image.rotate(skewing_angle if not random_skew else random_angle, expand=1)

        rc = (image.size[0], image.size[1])
        rotated_bboxes = rotate_bboxes(bboxes, skewing_angle if not random_skew else random_angle, rc)

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img # Mind = blown
        elif distorsion_type == 1:
            distorted_img = distorsion_generator.sin(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        elif distorsion_type == 2:
            distorted_img = distorsion_generator.cos(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        else:
            distorted_img = distorsion_generator.random(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            ratio = (float(size - vertical_margin) / float(distorted_img.size[1]))
            new_width = int(distorted_img.size[0] * ratio)
            resized_bboxes = resize_bboxes(rotated_bboxes, ratio, vertical_margin, horizontal_margin)
            resized_img = distorted_img.resize((new_width, size - vertical_margin), Image.ANTIALIAS)
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            ratio = (float(size - horizontal_margin) / float(distorted_img.size[0]))
            new_height = int(float(distorted_img.size[1]) * ratio)
            resized_img = distorted_img.resize((size - horizontal_margin, new_height), Image.ANTIALIAS)
            resized_bboxes = resize_bboxes(rotated_bboxes, ratio, vertical_margin, horizontal_margin)
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background = background_generator.gaussian_noise(background_height, background_width)
        elif background_type == 1:
            background = background_generator.plain_white(background_height, background_width)
        elif background_type == 2:
            background = background_generator.quasicrystal(background_height, background_width)
        else:
            background = background_generator.picture(background_height, background_width)

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background.paste(resized_img, (margin_left, margin_top), resized_img)
        elif alignment == 1:
            background.paste(resized_img, (int(background_width / 2 - new_text_width / 2), margin_top), resized_img)
        else:
            background.paste(resized_img, (background_width - new_text_width - margin_right, margin_top), resized_img)

        ##################################
        # Apply gaussian blur #
        ##################################

        final_image = background.filter(
            ImageFilter.GaussianBlur(
                radius=(blur if not random_blur else rnd.randint(0, blur))
            )
        )

        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = '{}_{}.{}'.format(text, str(index), extension)
        elif name_format == 1:
            image_name = '{}_{}.{}'.format(str(index), text, extension)
        elif name_format == 2:
            image_name = '{}.{}'.format(str(index),extension)
        else:
            print('{} is not a valid name format. Using default.'.format(name_format))
            image_name = '{}_{}.{}'.format(text, str(index), extension)

        # Save the image
        final_image = final_image.convert('RGB')
        if is_bbox:
            pdraw = ImageDraw.Draw(final_image)
            for bbox in resized_bboxes:
                pdraw.line(bbox+[bbox[0]], fill='red', width=2)

        resultDict = {
            "fn" : image_name,
            'charBB': None, 
            'txt': text
        }
        for i, box in enumerate(resized_bboxes):  # box : [lx,ly, rx,ly, rx,ry, lx,ry, '###']
            charBB = np.array(box).transpose()
            if i == 0:
                resultDict['charBB'] = np.dstack((charBB,))
            else:
                resultDict['charBB'] = np.dstack((resultDict['charBB'],charBB))

        if not label_only:
            with (Path(out_dir) / image_name).with_suffix('.pkl').open('wb') as pkl:
                if resultDict['charBB'] is None:
                    raise ValueError and "charBB is None"
                pickle.dump(resultDict, pkl)
        
        # with open(os.path.join(out_dir,'output.txt'), 'a', encoding='utf8') as outfile:
        #     outfile.write('{}\t{}\n'.format(image_name, text))
        
        final_image.save(Path(out_dir) / image_name)
