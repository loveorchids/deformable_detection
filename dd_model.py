import torch, sys, os, math
sys.path.append(os.path.expanduser("~/Documents"))
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torchvision.models import vgg16_bn
import omni_torch.networks.blocks as omth_blocks
import init
from dd_utils import *


cfg = {
    # Configuration for 512x512 input image
    'num_classes': 2,
    # Which conv layer output to use
    # The program create the prior box according to the length of conv_output
    # As long as its length does not exceed the length of other value
    # e.g. feature_map_sizes, box_height, box_height_large
    # Then it will be OK
    'conv_output': ["conv_4", "conv_5", "extra_2", "extra_3"],
    #'feature_map_sizes': [96, 48, 24, 24, 24],
    'feature_map_sizes': [64, 32, 16, 16, 16],
    # For static input size only, when Dynamic mode is turned out, it will not be used
    # Must be 2d list or tuple
    'input_img_size': [512, 512],
    #'input_img_size': [768, 768],
    # See the visualization result by enabling visualize_bbox in function fit of textbox.py
    # And change the settings according to the result
    # Some possible settings of box_height and box_height_large
    # 'box_height': [[16], [26], [36]],
    # 'box_height': [[10, 16], [26], [36]],
    # 'box_height': [[16], [26], []],
    'box_height': [[18], [30], [46], [68]],
    #'box_ratios': [[1, 2, 4, 7, 11, 15, 20, 26], [0.5, 1, 2, 5, 9, 13, 16, 18],
                   #[0.25, 0.5, 1, 2, 5, 8, 10], [0.5, 1, 2, 3, 5, 8]],
    'box_ratios': [[1, 4, 8, 13, 29, 26], [0.5, 1, 2, 5, 9, 15],
                   [1, 2, 5, 10, 15], [1, 2, 5, 8], [1, 2, 3, 4]],
    # If big_box is True, then box_height_large and box_ratios_large will be used
    'big_box': True,
    'box_height_large': [[24], [38], [56], [98]],
    #'box_ratios_large': [[0.5, 1, 2, 4, 7, 11, 15, 20], [0.3, 0.5, 1, 3, 6, 9, 11, 13],
                         #[0.25, 0.5, 1, 2, 4, 7, 9], [0.5, 1, 2, 3, 5]],
    'box_ratios_large': [[1, 2, 4, 7, 11, 15, 20], [0.5, 1, 3, 6, 9, 13],
                         [1, 2, 4, 7, 9], [1, 2, 3, 5], [1, 2, 3, 4]],
    # You can increase the stride when feature_map_size is large
    # especially at swallow conv layers, so as not to create lots of prior boxes
    'stride': [1, 1, 1, 1, 1],
    # Input depth for location and confidence layers
    'loc_and_conf': [512, 512, 256, 256, 384],
    # The hyperparameter to decide the Loss
    'variance': [0.1, 0.2],
    'var_updater': 1,
    'alpha': 1,
    'alpha_updater': 1,
    # Jaccard Distance Threshold
    'overlap_thresh': 0.45,
    # Whether to constrain the prior boxes inside the image
    'clip': True,
    'super_wide': 0.5,
    'super_wide_coeff': 0.5,
}

class FPN_block(nn.Module):
    def __init__(self, input_channel, output_channel, BN=nn.BatchNorm2d, upscale_factor=2):
        super().__init__()
        """
        module = []
        module.append(nn.Conv2d(in_channels=input_channel, kernel_size=3, padding=1,
                            out_channels=input_channel * upscale_factor * upscale_factor))
        module.append(nn.PixelShuffle(upscale_factor))
        self.up_conv = nn.Sequential(*module)
        self.norm_conv = omth_blocks.conv_block(input_channel, kernel_sizes=1,
                                                filters=output_channel, stride=1, padding=0,
                                                batch_norm=BN)
        """
        self.up_conv = omth_blocks.conv_block(input_channel, kernel_sizes=[2 + upscale_factor, 1],
                                              filters=[output_channel, output_channel], stride=[0 + upscale_factor, 1],
                                              padding=[1, 0], batch_norm=BN, transpose=[True, False])

    def forward(self, x):
        x = self.up_conv(x)
        #x = self.norm_conv(x)
        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class SSD(nn.Module):
    def __init__(self, cfg, btnk_chnl=512, batch_norm=nn.BatchNorm2d, fix_size=True,
                 connect_loc_to_conf=False, loc_incep=False, conf_incep=False,
                 loc_preconv=False, conf_preconv=False, nms_thres=0.2,
                 nms_top_k=1600, nms_conf_thres=0.01, FPN=False, SA=False, in_wid=64,
                 m_factor=1.0, with_extra4=True, extra_layer_setting=None):
        super().__init__()
        self.cfg = cfg
        self.FPN = FPN
        self.SA = SA
        self.num_classes = cfg['num_classes']
        self.output_list = cfg['conv_output']
        self.conv_module = nn.ModuleList([])
        self.loc_layers = nn.ModuleList([])
        self.conf_layers = nn.ModuleList([])
        self.conf_concate = nn.ModuleList([])
        self.fpn_back = nn.ModuleDict({})
        self.conv_module_name = []
        self.softmax = nn.Softmax(dim=-1)
        #self.detect = Detect(self.num_classes, 0, nms_top_k, nms_conf_thres, nms_thres)
        self.connect_loc_to_conf = connect_loc_to_conf
        self.fix_size = fix_size
        self.bottleneck_channel = btnk_chnl
        self.batch_norm = batch_norm
        self.with_extra4 = with_extra4
        self.extra_layer_filters = extra_layer_setting
        if fix_size:
            self.prior = self.create_prior()#.cuda()

        # Create the backbone model structure
        self.create_backbone_model()
        
        # Location and Confidence Layer
        for i  in range(len(cfg['conv_output'])):
            in_channel = cfg["loc_and_conf"][i]
            anchor = calculate_anchor_number(cfg, i)
            # Create Location and Confidence Layer
            self.loc_layers.append(
                self.create_loc_layer(in_channel, anchor, cfg['stride'][i], loc_incep=loc_incep,
                                      in_wid=in_wid, m_factor=m_factor, pre_layer=loc_preconv)
            )
            conf_layer, conf_concate = \
                self.create_conf_layer(in_channel, anchor, cfg['stride'][i], conf_incep=conf_incep,
                                       in_wid=in_wid, m_factor=m_factor, pre_layer=conf_preconv)
            self.conf_layers.append(conf_layer)
            self.conf_concate.append(conf_concate)


    def create_backbone_model(self):
        if self.extra_layer_filters is None:
            self.extra_layer_filters = [512, 384, 384, 256, 256, 256, 256, 256]
        # Prepare VGG-16 net with batch normalization
        vgg16_model = vgg16_bn(pretrained=True)
        net = list(vgg16_model.children())[0]
        # Replace the maxout with ceil in vanilla vgg16 net
        ceil_maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        net = [ceil_maxout if type(n) is nn.MaxPool2d else n for n in net]

        # Basic VGG Layers
        self.conv_module_name.append("conv_1")
        self.conv_module.append(nn.Sequential(*net[:6]))

        self.conv_module_name.append("conv_2")
        self.conv_module.append(nn.Sequential(*net[6:13]))

        self.conv_module_name.append("conv_3")
        self.conv_module.append(nn.Sequential(*net[13:23]))
        self.fpn_back.update({"conv_3": FPN_block(512, 256)})

        self.conv_module_name.append("conv_4")
        self.conv_module.append(nn.Sequential(*net[23:33]))
        self.fpn_back.update({"conv_4": FPN_block(512, 512)})

        self.conv_module_name.append("conv_5")
        self.conv_module.append(nn.Sequential(*net[33:43]))
        self.fpn_back.update({"conv_5": FPN_block(self.extra_layer_filters[1], 512, upscale_factor=1)})

        # Extra Layers
        self.conv_module_name.append("extra_1")
        self.conv_module.append(omth_blocks.conv_block(512, [self.extra_layer_filters[0], self.extra_layer_filters[1]],
                                                       kernel_sizes=[3, 3], stride=[1, 1], padding=[2, 3],
                                                       dilation=[2, 3], batch_norm=self.batch_norm))
        self.fpn_back.update({"extra_1": FPN_block(self.extra_layer_filters[3], self.extra_layer_filters[1], upscale_factor=2)})

        self.conv_module_name.append("extra_2")
        self.conv_module.append(omth_blocks.conv_block(self.extra_layer_filters[1], kernel_sizes=[3, 1],
                                                       filters=[self.extra_layer_filters[2], self.extra_layer_filters[3]],
                                                       stride=[1, 2], padding=[1, 0], batch_norm=self.batch_norm))
        self.fpn_back.update({"extra_2": FPN_block(self.extra_layer_filters[5], self.extra_layer_filters[3], upscale_factor=1)})

        self.conv_module_name.append("extra_3")
        self.conv_module.append(omth_blocks.conv_block(self.extra_layer_filters[3], kernel_sizes=[3, 1],
                                                       filters=[self.extra_layer_filters[4], self.extra_layer_filters[5]],
                                                       stride=[1, 1], padding=[1, 0], batch_norm=self.batch_norm))
        self.fpn_back.update({"extra_3": FPN_block(self.extra_layer_filters[7], self.extra_layer_filters[5], upscale_factor=1)})

        if self.with_extra4:
            self.conv_module_name.append("extra_4")
            self.conv_module.append(omth_blocks.conv_block(self.extra_layer_filters[5], kernel_sizes=[3, 1],
                                                           filters=[self.extra_layer_filters[6], self.extra_layer_filters[7]],
                                                           stride=[1, 1], padding=[1, 0], batch_norm=self.batch_norm))

        if self.SA:
            # self attention module
            if self.with_extra4:
                filter_num = self.extra_layer_filters[7]
            else:
                filter_num = self.extra_layer_filters[5]
            self.self_attn = Self_Attn(filter_num)

    def create_loc_layer(self, in_channel, anchor, stride, loc_incep=False, in_wid=64, m_factor=1.0,
                         pre_layer=True):
        loc_layer = nn.ModuleList([])
        if pre_layer:
            loc_layer.append(omth_blocks.conv_block(
                in_channel, [in_channel, in_channel], kernel_sizes=[3, 1], stride=[1, 1],
                padding=[2, 0], dilation=[2, 1], batch_norm=self.batch_norm))
        if loc_incep:
            wm = round(in_wid * m_factor)
            loc_layer.append(omth_blocks.InceptionBlock(in_channel,
                filters=[[wm, wm, in_wid, in_wid], [wm, in_wid, in_wid], [wm, wm, in_wid, in_wid]],
                kernel_sizes=[[[1, 7], [1, 3], 3, 1], [[1, 5], 3, 1], [[1, 3], [3, 1], 3, 1]],
                stride=[[1, 1, 1, 1], [1, 1, 1], [1, 1, 1, 1]],
                padding=[[[0, 3], [0, 1], 1, 0], [[0, 2], 1, 0], [[0, 1], [1, 0], 1, 0]],
                batch_norm=None, inner_maxout=None)
            )
        input_channel = in_wid * 3 if loc_incep else in_channel
        loc_layer.append(omth_blocks.conv_block(
            input_channel, filters=[input_channel, int(input_channel / 2), anchor * 4],
            kernel_sizes=[3, 3, 1], stride=[1, 1, stride], padding=[1, 1, 0], activation=None)
        )
        loc_layer.apply(init.init_cnn)
        return loc_layer

    def create_conf_layer(self, in_channel, anchor, stride, conf_incep=False, in_wid=64, m_factor=1.0,
                          pre_layer=True):
        conf_layer = nn.ModuleList([])
        if pre_layer:
            conf_layer.append(omth_blocks.conv_block(
                in_channel, [in_channel, in_channel], kernel_sizes=[3, 1], stride=[1, 1],
                padding=[3, 0], dilation=[3, 1], batch_norm=self.batch_norm)
            )
        if self.connect_loc_to_conf:
            if conf_incep:
                wm = round(in_wid * m_factor)
                out_chnl_2 = in_channel - (3 * in_wid)
                conf_layer.append(omth_blocks.InceptionBlock(
                    in_channel, stride=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1]],
                    kernel_sizes=[[[1, 7], 3, 1], [[1, 5], 3, 1], [[1, 3], 3, 1], [3, 1]],
                    filters=[[wm, in_wid, in_wid], [wm, in_wid, in_wid],
                             [wm, in_wid, in_wid], [round(out_chnl_2 * m_factor), out_chnl_2]],
                    padding=[[[0, 3], 1, 0], [[0, 2], 1, 0], [[0, 1], 1, 0], [1, 0]],
                    batch_norm=None, inner_maxout=None))
            else:
                conf_layer.append(omth_blocks.conv_block(
                    in_channel, filters=[in_channel, in_channel],
                    kernel_sizes=[1, 3], stride=[1, 1], padding=[0, 1], activation=None))
            # In this layer, the output from loc_layer will be concatenated to the conf layer
            # Feeding the conf layer with regressed location, helping the conf layer
            # to get better prediction
            conf_concate = omth_blocks.conv_block(
                in_channel + anchor * 4, kernel_sizes=[3, 3, 1],
                filters=[int(in_channel), int(in_channel / 4), anchor * 2],
                stride=[1, 1, stride], padding=[1, 1, 0], activation=None)
            conf_concate.apply(init.init_cnn)
        else:
            if conf_incep:
                wm = round(in_wid * m_factor)
                out_chnl_2 = in_channel - (3 * in_wid)
                conf_layer.append(omth_blocks.InceptionBlock(
                    in_channel, stride=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1]],
                    kernel_sizes=[[[1, 7], 3, 1], [[1, 5], 3, 1], [[1, 3], 3, 1], [3, 1]],
                    filters=[[wm, in_wid, in_wid], [wm, in_wid, in_wid],
                             [wm, in_wid, in_wid], [round(out_chnl_2 * m_factor), out_chnl_2]],
                    padding=[[[0, 3], 1, 0], [[0, 2], 1, 0], [[0, 1], 1, 0], [1, 0]],
                    batch_norm=None, inner_maxout=None))
            else:
                #print("conf_incep is turned off due to connect_loc_to_conf is False")
                conf_layer.append(omth_blocks.conv_block(
                    in_channel, filters=[in_channel, int(in_channel / 2), anchor * 2],
                    kernel_sizes=[3, 3, 1], stride=[1, 1, stride], padding=[1, 1, 0], activation=None))
            conf_concate = None
        conf_layer.apply(init.init_cnn)
        return conf_layer, conf_concate

    def create_prior(self, feature_map_size=None, input_size=None):
        """
        :param feature_map_size:
        :param input_size: When input size is not None. which means Dynamic Input Size
        :return:
        """
        from itertools import product as product
        mean = []
        big_box = self.cfg['big_box']
        if feature_map_size is None:
            assert len(self.cfg['feature_map_sizes']) >= len(self.cfg['conv_output'])
            feature_map_size = self.cfg['feature_map_sizes']
        if input_size is None:
            input_size = cfg['input_img_size']
        assert len(input_size) == 2, "input_size should be either int or list of int with 2 elements"
        input_ratio = input_size[1] / input_size[0]
        for k in range(len(self.cfg['conv_output'])):
            # Get setting for prior creation from cfg
            h, w = get_parameter(feature_map_size[k])
            h_stride, w_stride = get_parameter(cfg['stride'][k])
            for i, j in product(range(0, int(h), int(h_stride)), range(0, int(w), int(w_stride))):
                # 4 point represent: center_x, center_y, box_width, box_height
                cx = (j + 0.5) / w
                cy = (i + 0.5) / h
                # Add prior boxes with different height and aspect-ratio
                for height in self.cfg['box_height'][k]:
                    s_k = height / input_size[0]
                    for box_ratio in self.cfg['box_ratios'][k]:
                        mean += [cx, cy, s_k * box_ratio, s_k]
                # Add prior boxes with different number aspect-ratio if the box is large
                if big_box:
                    for height in self.cfg['box_height_large'][k]:
                        s_k_big = height / input_size[0]
                        for box_ratio_l in self.cfg['box_ratios_large'][k]:
                            mean += [cx, cy, s_k_big * box_ratio_l, s_k_big]
        # back to torch land
        prior_boxes = torch.Tensor(mean).view(-1, 4)
        if self.cfg['clip']:
            #boxes = center_size(prior_boxes, input_ratio)
            prior_boxes.clamp_(max=1, min=0)
            #prior_boxes = point_form(boxes, input_ratio)
        return prior_boxes

    def forward(self, x, is_train=True, verbose=False):
        input_size = [x.size(2), x.size(3)]
        locations, confidences, conv_output = [], [], []
        feature_shape = []
        features = []
        for i, conv_layer in enumerate(self.conv_module):
            x = conv_layer(x)
            if verbose:
                print("%s output shape: %s"%(self.conv_module_name[i], str(x.shape)))
            if i == len(self.conv_module) - 1 and self.SA:
                # apply self attention to the last convolutional layer output
                x, attn_map = self.self_attn(x)
            features.append(x)
        # remove the last one
        conv_output = []
        for i in range(len(features)):
            # idx is the reverse order of feature
            idx = len(features) - i - 1
            if self.FPN:
                if idx > 0:
                    key = self.conv_module_name[idx - 1]
                    #print(key)
                    if key in self.fpn_back:
                        x = self.fpn_back[key](features[idx]) + features[idx - 1]
                else:
                    continue
            else:
                x = features[idx - 1]
            if self.conv_module_name[idx - 1] in self.output_list:
                conv_output.append(x)
                # Get shape from each conv output so as to create prior
                feature_shape.append((x.size(2), x.size(3)))
                if verbose:
                    print("CNN output shape: %s" % (str(x.shape)))
            if len(conv_output) == len(self.output_list) and not self.FPN:
                # Doesn't need to compute further convolutional output
                break
        conv_output.reverse()
        if not self.fix_size:
            feature_shape.reverse()
            self.prior = self.create_prior(feature_map_size=feature_shape, input_size=input_size).cuda()
        for i, x in enumerate(conv_output):
            # Calculate location regression
            loc = x
            for layer in self.loc_layers[i]:
                loc = layer(loc)
            locations.append(loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4))
            # Calculate prediction confidence
            conf = x
            for layer in self.conf_layers[i]:
                conf = layer(conf)
            if self.connect_loc_to_conf:
                _loc = loc.detach()
                conf = torch.cat([conf, _loc], dim=1)
                for layer in self.conf_concate[i]:
                    conf = layer(conf)
            confidences.append(conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.num_classes))
            if verbose:
                print("Loc output shape: %s\nConf output shape: %s" % (str(loc.shape), str(conf.shape)))

        # Generate the result
        locations = torch.cat(locations, dim=1)
        confidences = torch.cat(confidences, dim=1)
        if is_train:
            #output = [locations, confidences, self.prior]
            return [locations, confidences, self.prior]
        else:
            #output = self.detect(locations, self.softmax(confidences), self.prior)
            return [locations, self.softmax(confidences), self.prior]
        #return output


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output.cuda()


if __name__ == "__main__":
    import time
    tmp = torch.randn(1, 3, 512, 512).to("cuda")
    x = torch.randn(6, 3, 512, 512).to("cuda")
    ssd = SSD(cfg, connect_loc_to_conf=True, conf_incep=True, loc_incep=True,
              FPN=True, SA=True, in_wid=48, m_factor=1.5, with_extra4=False).to("cuda")

    # Warm up
    _ = ssd(tmp)
    print("start")
    start = time.time()
    loc, conf, prior = ssd(x, verbose=True)
    print(loc.shape)
    print(conf.shape)
    print(prior.shape)
    print("Calculation cost: %s seconds"%(time.time() - start))
