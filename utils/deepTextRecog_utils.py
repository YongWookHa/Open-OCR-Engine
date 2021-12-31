import torch


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, tokens, is_character):
        # token (str): set of the possible tokens.
        self.dict = {}
        self.is_character = is_character
        for i, token in enumerate(tokens):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[token] = i + 1

        self.tokens = ['[CTCblank]'] + tokens  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, max_seq_len=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            max_seq_len: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, max_seq_len]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), max_seq_len).fill_(0)
        for i, t in enumerate(text):
            text = list(t) if self.is_character else t.split(' ')
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated tokens and blank.
                    char_list.append(self.tokens[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, is_character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.tokens = list_token + list_character

        self.is_character = is_character
        self.dict = {}
        for i, char in enumerate(self.tokens):
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
            text = list(t) if self.is_character else t.split(' ')
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.tokens[i] for i in text_index[index, :]])
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
