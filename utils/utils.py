import torch
import cv2
import base64
import numpy as np
import yaml

from statistics import median
from easydict import EasyDict


def load_setting(setting):
    with open(setting, 'r', encoding='utf8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(cfg)

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, tokens):
        # character (str): set of the possible characters.
        dict_character = list(tokens)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, max_seq_len=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            max_seq_len: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, max_seq_len]
            length: length of each text. [batch_size]
        """
        device = text.device
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), max_seq_len).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, max_seq_len=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, is_character=None):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, max_seq_len=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            max_seq_len: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # max_seq_len = max(length) # this is not allowed for multi-gpu setting
        max_seq_len += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), max_seq_len + 1).fill_(0)
        for i, t in enumerate(text):
            # text = t.split(' ')
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def decode_image_from_string(base64input):
    # "data:image/jpeg;base64,"
    header, image_string = base64input.split("base64,")
    ext = header.split('/')[1][:-1]
    nparr = np.frombuffer(base64.decodebytes(image_string.encode('utf8')), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    # return base64.decodebytes(byte_image.encode('utf8'))
    
def sort_wordBoxes(boxes) -> dict:
    '''
    boxes : [[lx, ly, rx, ry], [lx, ly, rx, ry], ...]
    '''
    
    y_sorted = sorted(boxes, key=lambda box: (box[1]+box[3])/2)
    prev_ry, char_height = y_sorted[0][3], median([box[3]-box[1] for box in y_sorted])
    y_diff = 0.0
    line_num, order = 0, 0
    lines = [[]]
    for box in y_sorted:
        y_diff = abs((box[3]+box[1])/2-prev_ry)
        prev_ry = (box[3]+box[1])/2
        if y_diff < char_height * 0.7:  # line_gap : 0.7
            order += 1
            lines[-1].append(box)
        else:
            line_num += 1
            order = 0
            lines.append([box])
    
    ret = []
    for line in lines:
        ret += sorted(line, key=lambda box: (box[0]+box[2])/2)
    
    return ret

def get_vocab(fn):
    with open(fn, 'r', encoding='utf8') as f:
        vocab = f.readline().strip().split()
    return vocab