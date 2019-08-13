import torch, sys, os, math
sys.path.append(os.path.expanduser("~/Documents"))
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torchvision.models import vgg16_bn
import omni_torch.networks.blocks as omth_blocks
import init
from dd_utils import *
from dd_model_cfg import *
import mmdet.ops.dcn as dcn
from itertools import product as product

cfg = ssd_512

class DetectionHeader(nn.Module):
    def __init__(self, in_channel, ratios, num_classes, opt, ):
        super().__init__()
        self.kernel_wise_deform = opt.kernel_wise_deform
        self.deformation_source = opt.deformation_source
        self.kernel_size = 3
        self.deformation = opt.deformation
        self.img_size = opt.img_size

        self.loc_layers = nn.ModuleList([])
        for i in range(ratios):
            self.loc_layers.append(nn.Conv2d(in_channel, 4, kernel_size=3, padding=1))

        if opt.deformation and opt.deformation_source.lower() not in ["geometric", "geometric_v2"]:
            self.offset_groups = nn.ModuleList([])
            if opt.deformation_source.lower() == "input":
                # Previous version, represent deformation_source is True
                offset_in_channel = in_channel
            elif opt.deformation_source.lower() == "regression":
                # Previous version, represent deformation_source is False
                offset_in_channel = 4
            elif opt.deformation_source.lower() == "concate":
                offset_in_channel = in_channel + 4
            else:
                raise NotImplementedError()
            if opt.kernel_wise_deform:
                deform_depth = 2
            else:
                deform_depth = 2 * (self.kernel_size ** 2)
            for i in range(ratios):
                pad = int(0.5 * (self.kernel_size - 1) + opt.deform_offset_dilation - 1)
                _offset2d = nn.Conv2d(offset_in_channel, deform_depth, kernel_size=self.kernel_size,
                                      bias=opt.deform_offset_bias, padding=pad,
                                      dilation=opt.deform_offset_dilation)
                self.offset_groups.append(_offset2d)

        self.conf_layers = nn.ModuleList([])
        for i in range(ratios):
            if opt.deformation:
                _deform = dcn.DeformConv(in_channel, num_classes, kernel_size=self.kernel_size, padding=1, bias=False)
            else:
                _deform = nn.Conv2d(in_channel, num_classes, kernel_size=3, padding=1)
            self.conf_layers.append(_deform)

    def forward(self, x, h, verbose=False, deform_map=False, priors=None, centeroids=None, cfg=None):
        # regression is a list, the length of regression equals to the number different aspect ratio
        # under current receptive field, elements of regression are PyTorch Tensor, encoded in
        # point-form, represent the regressed prior boxes.
        regression = [loc(x) for loc in self.loc_layers]
        if verbose:
            print("regression shape is composed of %d %s" % (len(regression), str(regression[0].shape)))
        if self.deformation:
            if self.deformation_source.lower() == "input":
                _deform_map = [offset(x) for offset in self.offset_groups]
            elif self.deformation_source.lower() == "regression":
                _deform_map = [offset(regression[i]) for i, offset in enumerate(self.offset_groups)]
            elif self.deformation_source.lower() in ["geometric", "geometric_v2"]:
                _deform_map = []
                for i, reg in enumerate(regression):
                    # get the index of certain ratio from prior box
                    idx = torch.tensor([i + len(regression) * _ for _ in range(reg.size(2) * reg.size(3))]).long()
                    prior = priors[idx, :]
                    prior_center = centeroids[idx, :].repeat(x.size(0), 1)
                    _reg = decode(reg.permute(0, 2, 3, 1).contiguous().view(-1, 4),
                                  prior.repeat(x.size(0), 1), cfg["variance"]).clamp(min=0, max=1)
                    reg_center = center_conv_point(_reg)
                    # print(_reg[0, :].data, point_form(prior[0:1, :]).clamp(min=0, max=1).data)
                    # TODO: In the future work, when input image is not square, we need
                    # TODO: to multiply image with its both width and height
                    df_map = (reg_center - prior_center) * x.size(2)
                    _deform_map.append(df_map.view(x.size(0), reg.size(2), reg.size(3), -1)
                                       .permute(0, 3, 1, 2))
            elif self.deformation_source.lower() == "concate":
                # TODO: reimplement forward graph
                raise NotImplementedError()
            else:
                raise NotImplementedError()

            if verbose:
                print("deform_map shape is composed of %d %s" % (len(_deform_map), str(_deform_map[0].shape)))
            if self.kernel_wise_deform:
                _deform_map = [dm.repeat(1, self.kernel_size ** 2, 1, 1) for dm in _deform_map]
            # Amplify the offset signal, so it can deform the kernel to adjacent anchor
            #_deform_map = [dm * h/x.size(2) for dm in _deform_map]
            if verbose:
                print("deform_map shape is extended to %d %s" % (len(_deform_map), str(_deform_map[0].shape)))
            pred = [deform(x, _deform_map[i]) for i, deform in enumerate(self.conf_layers)]
        else:
            pred = [conf(x) for conf in self.conf_layers]
            _deform_map = None
        if verbose:
            print("pred shape is composed of %d %s" % (len(pred), str(pred[0].shape)))
        if deform_map:
            return torch.cat(regression, dim=1), torch.cat(pred, dim=1), _deform_map
        else:
            return torch.cat(regression, dim=1), torch.cat(pred, dim=1)

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
        return x


class SSD(nn.Module):
    def __init__(self, cfg, btnk_chnl=512, batch_norm=nn.BatchNorm2d, fix_size=True,
                 connect_loc_to_conf=False, loc_incep=False, conf_incep=False,
                 loc_preconv=False, conf_preconv=False, FPN=False, SA=False, in_wid=64,
                 m_factor=1.0, extra_layer_setting=None):
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
        self.conv_module.append(nn.Sequential(*net[:7]))

        self.conv_module_name.append("conv_2")
        self.conv_module.append(nn.Sequential(*net[7:14]))

        self.conv_module_name.append("conv_3")
        self.conv_module.append(nn.Sequential(*net[14:24]))
        self.fpn_back.update({"conv_3": FPN_block(512, 256)})

        self.conv_module_name.append("conv_4")
        self.conv_module.append(nn.Sequential(*net[24:34]))
        self.fpn_back.update({"conv_4": FPN_block(512, 512)})

        self.conv_module_name.append("conv_5")
        self.conv_module.append(nn.Sequential(*net[34:44]))
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
        self.fpn_back.update({"extra_2": FPN_block(self.extra_layer_filters[5], self.extra_layer_filters[3], upscale_factor=2)})

        self.conv_module_name.append("extra_3")
        self.conv_module.append(omth_blocks.conv_block(self.extra_layer_filters[3], kernel_sizes=[3, 1],
                                                       filters=[self.extra_layer_filters[4], self.extra_layer_filters[5]],
                                                       stride=[1, 2], padding=[1, 0], batch_norm=self.batch_norm))
        self.fpn_back.update({"extra_3": FPN_block(self.extra_layer_filters[7], self.extra_layer_filters[5], upscale_factor=2)})

        self.conv_module_name.append("extra_4")
        self.conv_module.append(omth_blocks.conv_block(self.extra_layer_filters[5], kernel_sizes=[3, 1],
                                                       filters=[self.extra_layer_filters[6], self.extra_layer_filters[7]],
                                                       stride=[1, 2], padding=[1, 0], batch_norm=self.batch_norm))


    def create_prior(self, feature_map_size=None, input_size=None):
        """
        :param feature_map_size:
        :param input_size: When input size is not None. which means Dynamic Input Size
        :return:
        """

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
            idx = len(features) - i# - 1
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
              FPN=False, SA=False, in_wid=48, m_factor=1.5).to("cuda")

    # Warm up
    _ = ssd(tmp)
    print("start")
    start = time.time()
    loc, conf, prior = ssd(x, verbose=True)
    print("loc shape: %s"%str(loc.shape))
    print("conf shape: %s" % str(conf.shape))
    print("prior shape: %s" % str(prior.shape))
    assert loc.size(1) == conf.size(1) == prior.size(0)
    print("Calculation cost: %s seconds"%(time.time() - start))
