import torch
from torch import nn
from torch.nn import functional as F

from models.base import ModelBase
from models.craft_modules import init_weights, double_conv, vgg16_bn

from utils.craft_utils import hard_negative_mining
from utils.misc import calculate_batch_fscore, generate_word_bbox_batch


class CRAFT(ModelBase):
    def __init__(self, cfg, pretrained=False, freeze=False):
        super(CRAFT, self).__init__(cfg)
        self.cfg = cfg

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature

    def training_step(self, batch, batch_nb):
        big_image, weight, weight_affinity = batch

        output, _ = self(big_image)

        loss = self.cal_loss(output, weight, weight_affinity)
        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_num):
        big_image, weight, weight_affinity = batch

        output, _ = self(big_image)
        loss = self.cal_loss(output, weight, weight_affinity)

        if type(output) == list:
            output = torch.cat(output, dim=0)

        predicted_bbox = generate_word_bbox_batch(
            output[:, :, :, 0].data.cpu().numpy(),
            output[:, :, :, 1].data.cpu().numpy(),
            character_threshold=self.cfg.craft.THRESHOLD_CHARACTER,
            affinity_threshold=self.cfg.craft.THRESHOLD_AFFINITY,
            word_threshold=self.cfg.craft.THRESHOLD_WORD,
        )

        target_bbox = generate_word_bbox_batch(
            weight.data.cpu().numpy(),
            weight_affinity.data.cpu().numpy(),
            character_threshold=self.cfg.craft.THRESHOLD_CHARACTER,
            affinity_threshold=self.cfg.craft.THRESHOLD_AFFINITY,
            word_threshold=self.cfg.craft.THRESHOLD_WORD
        )

        fscore, precision, recall = calculate_batch_fscore(
            predicted_bbox,
            target_bbox,
            threshold=self.cfg.craft.THRESHOLD_FSCORE,
            text_target=None
        )

        return {'val_loss': loss,
                'fscore':fscore,
                'precision' :precision,
                'recall':recall}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        fscore = sum([x['fscore'] for x in outputs]) / len(outputs)
        precision = sum([x['precision'] for x in outputs]) / len(outputs)
        recall = sum([x['recall'] for x in outputs]) / len(outputs)

        self.log('val_loss', val_loss)
        self.log('fscore', fscore)
        self.log('precision', precision)
        self.log('recall', recall)

    def cal_loss(self, output, character_map, affinity_map):
        """
        :param output: prediction output of the model of shape [batch_size, 2, height, width]
        :param character_map: target character map of shape [batch_size, height, width]
        :param affinity_map: target affinity map of shape [batch_size, height, width]
        :return: loss containing loss of character heat map and affinity heat map reconstruction
        """

        batch_size, height, width, channels = output.shape

        output = output.contiguous().view([batch_size * height * width, channels])

        character = output[:, 0]
        affinity = output[:, 1]

        affinity_map = affinity_map.view([batch_size * height * width])
        character_map = character_map.view([batch_size * height * width])

        loss_character = hard_negative_mining(character, character_map, self.cfg)
        loss_affinity = hard_negative_mining(affinity, affinity_map, self.cfg)

        # weight character twice then affinity
        all_loss = loss_character * 2 + loss_affinity

        return all_loss
    
    def predict(self, image):
        """
        input:
            image: (np.ndarray) [H, W, C] 
        out:
            list of word boxes
        """
        img, (h_pad, w_pad), ratio = self.resize(image, self.cfg.img_size)
        x = self.transform(img).unsqueeze(0).to(self.device)
        out, _ = self(x)
        
        predicted_bbox = generate_word_bbox_batch(
            out[:, :, :, 0].data.cpu().numpy(),
            out[:, :, :, 1].data.cpu().numpy(),
            character_threshold=self.cfg.craft.THRESHOLD_CHARACTER,
            affinity_threshold=self.cfg.craft.THRESHOLD_AFFINITY,
            word_threshold=self.cfg.craft.THRESHOLD_WORD,
        )
        
        boxes = predicted_bbox[0][:,[0,2],:,:].squeeze(2)
        boxes[:, :, 0] -= w_pad
        boxes[:, :, 1] -= h_pad
        total_boxes = boxes.reshape(-1,4) / ratio
        results = total_boxes.astype(int).tolist()
        
        return results