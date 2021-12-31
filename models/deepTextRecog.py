import pytorch_lightning as pl
import numpy as np
import torch

import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from models.base import ModelBase
from models.transformation import TPS_SpatialTransformerNetwork
from models.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from models.sequence_modeling import BidirectionalLSTM
from models.prediction import Attention

from utils.deepTextRecog_utils import CTCLabelConverter, AttnLabelConverter

class DeepTextRecog(ModelBase):
    def __init__(self, cfg, tokens):
        super(DeepTextRecog, self).__init__(cfg)

        if 'CTC' in cfg.dtr.Prediction:
            self.converter = CTCLabelConverter(tokens, cfg.dtr.is_character)
            self.criterion = torch.nn.CTCLoss(zero_infinity=True)
        else:
            self.converter = AttnLabelConverter(tokens, cfg.dtr.is_character)
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        cfg.dtr.num_class = len(self.converter.tokens)

        cfg.dtr.update({
            'num_fiducial': 20,
            'output_channel': 512,
            'hidden_size': 256,
        })

        self.cfg = cfg
        self.stages = {'Trans': cfg.dtr.Transformation, 'Feat': cfg.dtr.FeatureExtraction,
                       'Seq': cfg.dtr.SequenceModeling, 'Pred': cfg.dtr.Prediction}

        """ Transformation """
        if cfg.dtr.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=cfg.dtr.num_fiducial,
                I_size=(cfg.dtr.imgH, cfg.dtr.imgW),
                I_r_size=(cfg.dtr.imgH, cfg.dtr.imgW),
                I_channel_num=cfg.dtr.input_channel,
            )
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if cfg.dtr.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(cfg.dtr.input_channel, cfg.dtr.output_channel)
        elif cfg.dtr.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(cfg.dtr.input_channel, cfg.dtr.output_channel)
        elif cfg.dtr.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(cfg.dtr.input_channel, cfg.dtr.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = cfg.dtr.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = torch.nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if cfg.dtr.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = torch.nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, cfg.dtr.hidden_size, cfg.dtr.hidden_size),
                BidirectionalLSTM(cfg.dtr.hidden_size, cfg.dtr.hidden_size, cfg.dtr.hidden_size))
            self.SequenceModeling_output = cfg.dtr.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if cfg.dtr.Prediction == 'CTC':
            self.Prediction = torch.nn.Linear(self.SequenceModeling_output, cfg.dtr.num_class)
        elif cfg.dtr.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, cfg.dtr)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=False):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(),
                                         text.to(self.device),
                                         is_train,
                                         max_seq_len=self.cfg.dtr.max_seq_len)

        return prediction

    def get_text(self, preds):
        batch_size = preds.size(0)
        length_for_pred = [self.cfg.dtr.max_seq_len]*preds.size(0)

        texts = []
        if 'CTC' in self.cfg.dtr.Prediction:

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        texts = []
        for order, tup in enumerate(zip(preds_str, preds_max_prob)):
            pred, pred_max_prob = tup
            if 'Attn' in self.cfg.dtr.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                texts.append(pred)
                # pred_max_prob = pred_max_prob[:pred_EOS]

                ''' you can use confidence scroe '''
                # try:
                #     confidence_score = float(pred_max_prob.cumprod(dim=0)[-1])
                # except IndexError:
                #     confidence_score = 0.1

        return texts

    # def configure_optimizers(self):
    #     filtered_parameters = []
    #     params_num = []
    #     for p in filter(lambda p: p.requires_grad, self.parameters()):
    #         filtered_parameters.append(p)
    #         params_num.append(np.prod(p.size()))
    #     print('Trainable params num : ', sum(params_num))

    #     optimizer = torch.optim.Adadelta(filtered_parameters,
    #                                     lr=self.cfg.lr,
    #                                     rho=0.95,
    #                                     eps=1e-8)
    #     return optimizer

    def training_step(self, batch, batch_num):
        image_tensors, labels = batch
        image = image_tensors.to(self.device)
        text, length = self.converter.encode(
            labels, max_seq_len=self.cfg.dtr.max_seq_len)
        batch_size = image.size(0)

        if 'CTC' in self.cfg.dtr.Prediction:
            preds = self(image, text, is_train=True)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = self.criterion(preds, text, preds_size, length)

        else:
            # align with Attention.forward
            preds = self(image, text[:, :-1], is_train=True)
            target = text[:, 1:].to(self.device)  # without [GO] Symbol
            cost = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        self.log('train_loss', cost)

        return {'loss': cost}

    def validation_step(self, batch, batch_num):
        image_tensors, labels = batch
        image = image_tensors

        batch_size = image_tensors.size(0)

        # For max length prediction
        length_for_pred = torch.IntTensor([self.cfg.dtr.max_seq_len] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.cfg.dtr.max_seq_len + 1).fill_(0).to(self.device)

        text_for_loss, length_for_loss = self.converter.encode(labels,
                                    max_seq_len=self.cfg.dtr.max_seq_len)

        if 'CTC' in self.cfg.dtr.Prediction:
            preds = self(image, text_for_pred)

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = self.criterion(preds.log_softmax(2).permute(1, 0, 2),
                                  text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self(image, text_for_pred, is_train=False)

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:].to(self.device)  # without [GO] Symbol
            cost = self.criterion(preds.contiguous().view(-1, preds.shape[-1]),
                                  target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        n_correct = 0
        norm_ED = 0

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in self.cfg.dtr.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

        acc = n_correct / float(batch_size)
        norm_ED = norm_ED / float(batch_size)

        return {'val_loss': cost, 'acc': acc, 'norm_ED': norm_ED}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        acc = sum([x['acc'] for x in outputs]) / len(outputs)
        norm_ED = sum([x['norm_ED'] for x in outputs]) / len(outputs)

        self.log('val_loss', val_loss)
        self.log('acc', acc)
        self.log('norm_ED', norm_ED)
