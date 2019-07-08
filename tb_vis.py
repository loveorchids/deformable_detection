import os, torch, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import omni_torch.visualize.basic as vb
from matplotlib import gridspec
from researches.ocr.textbox.tb_utils import *

def print_box(red_boxes=(), shape=0, green_boxes=(), blue_boxes=(), img=None,
              idx=None, title=None, step_by_step_r=False, step_by_step_g=False,
              step_by_step_b=False, name_prefix=None, save_dir=None):
    # Generate the save folder and image save name
    if not name_prefix:
        name_prefix = "tmp"
    if idx is not None:
        img_name = name_prefix + "_sample_%s_pred" % (idx)
    else:
        img_name = name_prefix
    if save_dir is None:
        save_dir = os.path.expanduser("~/Pictures")
    else:
        save_dir = os.path.expanduser(save_dir)
    if not os.path.exists(save_dir):
        warnings.warn(
            "The save_dir you specified (%s) does not exist, saving results under "
            "~/Pictures"%(save_dir)
        )
        save_dir = os.path.expanduser("~/Pictures")
    img_path = os.path.join(save_dir, img_name)

    # Figure out the shape
    if type(shape) is tuple:
        h, w = shape[0], shape[1]
    else:
        h, w = shape, shape
    # img as white background image
    if img is None:
        img = np.zeros((h, w, 3)).astype(np.uint8) + 254
    else:
        img = img.astype(np.uint8)
        h, w, c = img.shape

    # Perform Visualization of boundbox
    fig, ax = plt.subplots(figsize=(round(w / 100), round(h / 100)))
    ax.imshow(img)
    step = 0
    for box in red_boxes:
        x1, y1, x2, y2 = coord_to_rect(box, h, w)
        rect = patches.Rectangle((x1, y1), x2, y2, linewidth=1,
                                       edgecolor='r', facecolor='none', alpha=1)
        ax.add_patch(rect)
        if step_by_step_r:
            plt.savefig(img_path + "_red_step_%s.jpg"%(str(step).zfill(4)))
            step += 1
    for box in green_boxes:
        x1, y1, x2, y2 = coord_to_rect(box, h, w)
        rect = patches.Rectangle((x1, y1), x2, y2, linewidth=2,
                                       edgecolor='g', facecolor='none', alpha=0.4)
        ax.add_patch(rect)
        if step_by_step_g:
            plt.savefig(img_path + "_green_step_%s.jpg" % (str(step).zfill(4)))
            step += 1
    for box in blue_boxes:
        x1, y1, x2, y2 = coord_to_rect(box, h, w)
        rect = patches.Rectangle((x1, y1), x2, y2, linewidth=2,
                                       edgecolor='b', facecolor='none', alpha=0.4)
        ax.add_patch(rect)
        if step_by_step_b:
            plt.savefig(img_path + "_blue_step_%s.jpg" % (str(step).zfill(4)))
            step += 1
    if title:
        plt.title(title)
    plt.savefig(os.path.join(save_dir, img_name + ".jpg"))
    plt.close()
    
def visualize_overlaps(cfg, target, label, prior, ratio):
    images, subtitle, coords = [], [], []

    # conf中的1代表所有当前设置下与ground truth匹配的default box及其相应的index
    overlaps, conf = match(cfg, cfg['overlap_thresh'], target, prior,
                           None, label, None, None, 0, ratio, visualize=True)
    summary = "%s of %s positive samples"%(int(torch.sum(conf)), prior.size(0))
    crop_start = 0

    for k in range(len(cfg['conv_output'])):
        # Get the setting from cfg to calculate number of anchor and prior boxes
        h, w = get_parameter(cfg['feature_map_sizes'][k])
        h_stride, w_stride = get_parameter(cfg['stride'][k])
        anchor_num = calculate_anchor_number(cfg, k)
        prior_num = len(range(0, int(h), int(h_stride))) * \
                    len(range(0, int(w), int(w_stride))) * anchor_num

        # Get the index of matched prior boxes and collect these boxes
        _conf = conf[crop_start: crop_start + prior_num]
        _overlaps = overlaps[crop_start: crop_start + prior_num]
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        best_prior_idx = best_prior_idx.squeeze(1)
        
        #points = point_form(prior[best_prior_idx].cpu(), ratio)
        #print_box(target, shape=? blue_boxes=points, step_by_step_b=True)
        
        matched_priors = int(torch.sum(_conf))
        idx = _conf == 1
        idx = list(np.where(idx.cpu().numpy() == 1)[0])
        for i in idx:
            coords.append(point_form(prior[crop_start+i:crop_start+i+1, :], ratio).squeeze())

        # Reshape _conf into the shape of image so as to visualize it
        _conf = _conf.view(len(range(0, int(h), int(h_stride))),
                           len(range(0, int(w), int(w_stride))), anchor_num)
        _conf = _conf.permute(2, 0, 1)
        subs = ["ratio: %s"%(r) for r in cfg['box_ratios'][k]]
        subtitle.append("box height: %s\neffective samle: %s"
                        %(cfg['box_height'][k], matched_priors))
        if cfg['big_box']:
            subs += ["ratio: %s"%(r) for r in cfg['box_ratios_large'][k]]
            subtitle[-1] = "box height: %s and %s\neffective samle: %s" \
                           %(cfg['box_height'][k], cfg['box_height_large'][k], matched_priors)

        # Convert _conf into open-cv form
        image = vb.plot_tensor(None, _conf.unsqueeze_(1) * 254, deNormalize=False,
                               sub_title=subs)
        images.append(image.astype(np.uint8))
        crop_start += prior_num
    return images, summary, subtitle, coords


def visualize_bbox(args, cfg, images, targets, prior=None, idx=0):
    print("Visualizing bound box...")
    ratios = images.size(3) / images.size(2)
    batch = images.size(0)
    height, width = images.size(2) / 100 + 1, images.size(3) / 50 + 1
    for i in range(batch):
        image = images[i:i+1, :, :, :]
        bbox = targets[i]

        image = vb.plot_tensor(args, image, deNormalize=True, margin=0).astype("uint8")
        h, w = image.shape[0], image.shape[1]
        # Create a Rectangle patch
        rects = []
        for point in bbox:
            x1, y1, x2, y2 = coord_to_rect(point, h, w)
            rects.append(patches.Rectangle((x1, y1), x2, y2, linewidth=2,
                                           edgecolor='r', facecolor='none'))
        if prior is not None:
            overlaps, summary, subtitle, coords = \
                visualize_overlaps(cfg, bbox[:, :-1].data, bbox[:, -1].data, prior, ratios)
            for coord in coords:
                x1, y1, x2, y2 = coord_to_rect(coord, h, w)
                rects.append(patches.Rectangle((x1, y1), x2, y2, linewidth=1,
                                               edgecolor='b', facecolor='none', alpha=0.3))
        else:
            overlaps = []
            summary = ""
        fig, ax = plt.subplots(figsize=(width + len(overlaps), height))
        width_ratio = [2] + [1] * len(overlaps)
        gs = gridspec.GridSpec(1, 1+len(overlaps), width_ratios=width_ratio)
        ax0 = plt.subplot(gs[0])
        ax0.imshow(image)
        ax0.set_title(summary)
        for j in range(len(overlaps)):
            ax = plt.subplot(gs[j + 1])
            ax.imshow(overlaps[j])
            ax.set_title(subtitle[j])
        for rect in rects:
            ax0.add_patch(rect)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.log_dir, "batch_%s_sample_vis_%s.jpg"%(idx, i)))
        plt.close()

