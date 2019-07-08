import os, time, sys, math, random, glob, datetime, argparse

sys.path.append(os.path.expanduser("~/Documents"))
import cv2, torch
import numpy as np
import imgaug
from imgaug import augmenters
import omni_torch.utils as util
import dd_preset as preset
import dd_model as model
import dd_data as tb_data
from dd_utils import *
from dd_preprocess import *
from dd_augment import *
from dd_postprocess import combine_boxes
from dd_vis import visualize_bbox, print_box
import omni_torch.visualize.basic as vb

TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
cfg = model.cfg
args = util.get_args(preset.PRESET)
if not torch.cuda.is_available():
    raise RuntimeError("Need cuda devices")
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
# Image will be resize to this size
square = 2048


def parse_arguments():
    parser = argparse.ArgumentParser(description='Textbox Detector Settings')
    ##############
    #        TRAINING        #
    ##############
    parser.add_argument(
        "-did",
        "--device_id",
        type=int,
        help="1 represent the latest model",
        default=0
    )
    parser.add_argument(
        "-tdr",
        "--test_dataset_root",
        type=str,
        help="1 represent the latest model",
        default="~/Pictures/dataset/ocr/tempholding"
    )
    parser.add_argument(
        "-ext",
        "--extension",
        type=str,
        help="extention of image",
        default="png"
    )
    parser.add_argument(
        "-gt_ext",
        "--ground_truth_extension",
        type=str,
        help="ground truth extention (text file) of image",
        default="xml"
    )
    """
    parser.add_argument(
        "-mpl",
        "--model_prefix_list",
        nargs='+',
        help="a list of model prefix to do the ensemble",
        default=["32imf2x_noextra4_nol2c_00000"]
        # default=["th"]
        # this fit the current tb_model.py code: ["ft_003_3"]
    )
    """
    parser.add_argument(
        "-mpl",
        "--model_prefix_list",
        type=str,
        help="a list of model prefix to do the ensemble",
        # default=["th"]
        # this fit the current tb_model.py code: ["ft_003_3"]
    )
    parser.add_argument(
        "-nth",
        "--nth_best_model",
        type=int,
        help="1 represent the latest model",
        default=1
    )
    parser.add_argument(
        "-dtk",
        "--detector_top_k",
        type=int,
        help="get top_k boxes from prediction",
        default=2500
    )
    parser.add_argument(
        "-dct",
        "--detector_conf_threshold",
        type=float,
        help="detector_conf_threshold",
        default=0.05
    )
    parser.add_argument(
        "-dnt",
        "--detector_nms_threshold",
        type=float,
        help="detector_nms_threshold",
        default=0.3
    )

    parser.add_argument(
        "-if",
        "--inner_filters",
        type=int,
        help="filter numbers in loc and conf block",
        default=64
    )
    parser.add_argument(
        "-imf",
        "--inner_m_factor",
        type=float,
        help="multiplication factor for numbers in loc and conf block",
        default=1.0
    )
    parser.add_argument(
        "-fpn",
        "--feature_pyramid_net",
        action="store_true",
        help="use feature pyramid network",
    )
    parser.add_argument(
        "-sa",
        "--self_attention",
        action="store_true",
        help="use self attention",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show precision and recall for each image",
    )
    parser.add_argument(
        "-l2c",
        "--loc_to_conf",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--conf_incep",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--loc_incep",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--conf_preconv",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--loc_preconv",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    return args


def augment_back(height_ori, width_ori, v_crop, h_crop, transform_det=None):
    aug_list = []
    v_crop = round(v_crop)
    h_crop = round(h_crop)
    # counteract pading
    aug_list.append(
        # top, right, bottom, left
        augmenters.Crop(px=(v_crop, h_crop, v_crop, h_crop))
    )
    # counteract resizing
    aug_list.append(
        augmenters.Resize(size={"height": height_ori, "width": width_ori})
    )
    # counteract rotation, if exist
    if transform_det:
        if "rotation" in transform_det:
            aug_list.append(
                augmenters.Affine(rotate=-transform_det["rotation"], cval=args.aug_bg_color),
            )
    aug = augmenters.Sequential(aug_list, random_order=False)
    return aug


def test_rotation(opt):
    result_dir = os.path.join(args.path, args.code_name, "result+" + "-".join(opt.model_prefix_list))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # Load
    if type(opt.model_prefix_list) is str:
        opt.model_prefix_list = [opt.model_prefix_list]
    assert len(opt.model_prefix_list) <= torch.cuda.device_count(), \
        "number of models should not exceed the device numbers"
    nets = []
    for _, prefix in enumerate(opt.model_prefix_list):
        #net = model.SSD(cfg, connect_loc_to_conf=True, fix_size=False,
                        #conf_incep=True, loc_incep=True)
        net = model.SSD(cfg, connect_loc_to_conf=opt.loc_to_conf, fix_size=False,
                        conf_incep=opt.conf_incep, loc_incep=opt.loc_incep,
                        loc_preconv=opt.loc_preconv, conf_preconv=opt.conf_preconv,
                        FPN=opt.feature_pyramid_net, SA=opt.self_attention,
                        in_wid=opt.inner_filters, m_factor=opt.inner_m_factor)
        device_id = opt.device_id if len(opt.model_prefix_list) == 1 else _
        net = net.to("cuda:%d" % (device_id))
        net_dict = net.state_dict()
        weight_dict = util.load_latest_model(args, net, prefix=prefix,
                                             return_state_dict=True, nth=opt.nth_best_model)
        loading_fail_signal = False
        for i, key in enumerate(net_dict.keys()):
            if "module." + key not in weight_dict:
                net_dict[key] = torch.zeros(net_dict[key].shape)
        for key in weight_dict.keys():
            if key[7:] in net_dict:
                if net_dict[key[7:]].shape == weight_dict[key].shape:
                    net_dict[key[7:]] = weight_dict[key]
                else:
                    print("Key: %s from disk has shape %s copy to the model with shape %s" %
                          (key[7:], str(weight_dict[key].shape), str(net_dict[key[7:]].shape)))
                    loading_fail_signal = True
            else:
                print("Key: %s does not exist in net_dict" % (key[7:]))
        if loading_fail_signal:
            raise RuntimeError('Shape Error happens, remove "%s" from your -mpl settings.' % (prefix))

        net.load_state_dict(net_dict)
        net.eval()
        nets.append(net)
        print("Above model loaded with out a problem")
    detector = model.Detect(num_classes=2, bkg_label=0, top_k=opt.detector_top_k,
                            conf_thresh=opt.detector_conf_threshold, nms_thresh=opt.detector_nms_threshold)

    # Enumerate test folder
    root_path = os.path.expanduser(opt.test_dataset_root)
    if not os.path.exists(root_path):
        raise FileNotFoundError("%s does not exists, please check your -tdr/--test_dataset_root settings" % (root_path))
    img_list = glob.glob(root_path + "/*.%s" % (opt.extension))
    precisions, recalls = [], []
    for i, img_file in enumerate(sorted(img_list)):
        start = time.time()
        name = img_file[img_file.rfind("/") + 1: -4]
        img = cv2.imread(img_file)
        height_ori, width_ori = img.shape[0], img.shape[1]

        do_rotation = False
        if do_rotation:
            # detect rotation for returning the image back
            img, transform_det = estimate_angle(img, args, None, None, None)
            transform_det["rotation"] = 0
            if transform_det["rotation"] != 0:
                rot_aug = augmenters.Affine(rotate=transform_det["rotation"],
                                            cval=args.aug_bg_color)
            else:
                rot_aug = None

            # Perform Augmentation
            if rot_aug:
                rot_aug = augmenters.Sequential(
                    augmenters.Affine(rotate=transform_det["rotation"], cval=args.aug_bg_color))
                image = rot_aug.augment_image(img)
            else:
                image = img
        else:
            image = img
        # Resize the longer side to a certain length
        if height_ori >= width_ori:
            resize_aug = augmenters.Sequential([
                augmenters.Resize(size={"height": square, "width": "keep-aspect-ratio"})])
        else:
            resize_aug = augmenters.Sequential([
                augmenters.Resize(size={"height": "keep-aspect-ratio", "width": square})])
        resize_aug = resize_aug.to_deterministic()
        image = resize_aug.augment_image(image)
        h_re, w_re = image.shape[0], image.shape[1]
        # Pad the image into a square image
        pad_aug = augmenters.Sequential(
            augmenters.PadToFixedSize(width=square, height=square, pad_cval=255, position="center")
        )
        pad_aug = pad_aug.to_deterministic()
        image = pad_aug.augment_image(image)
        h_final, w_final = image.shape[0], image.shape[1]

        # Prepare image tensor and test
        image_t = torch.Tensor(util.normalize_image(args, image)).unsqueeze(0)
        image_t = image_t.permute(0, 3, 1, 2)
        # visualize_bbox(args, cfg, image, [torch.Tensor(rot_coord).cuda()], net.prior, height_final/width_final)

        text_boxes = []
        for _, net in enumerate(nets):
            device_id = opt.device_id if len(nets) == 1 else _
            image_t = image_t.to("cuda:%d" % (device_id))
            out = net(image_t, is_train=False)
            loc_data, conf_data, prior_data = out
            prior_data = prior_data.to("cuda:%d" % (device_id))
            det_result = detector(loc_data, conf_data, prior_data)
            # Extract the predicted bboxes
            idx = det_result.data[0, 1, :, 0] >= 0.1
            text_boxes.append(det_result.data[0, 1, idx, 1:])
        text_boxes = torch.cat(text_boxes, dim=0)
        text_boxes = combine_boxes(text_boxes, img=image_t)
        pred = [[float(coor) for coor in area] for area in text_boxes]
        BBox = [
            imgaug.augmentables.bbs.BoundingBox(box[0] * w_final, box[1] * h_final, box[2] * w_final, box[3] * h_final)
            for box in pred]
        BBoxes = imgaug.augmentables.bbs.BoundingBoxesOnImage(BBox, shape=image.shape)
        return_aug = augment_back(height_ori, width_ori, (h_final - h_re) / 2, (w_final - w_re) / 2)
        return_aug = return_aug.to_deterministic()
        img_ori = return_aug.augment_image(image)
        bbox = return_aug.augment_bounding_boxes([BBoxes])[0]
        # print_box(blue_boxes=pred, idx=i, img=vb.plot_tensor(args, image_t, margin=0),
        # save_dir=args.val_log)

        f = open(os.path.join(result_dir, name + ".txt"), "w")
        gt_box_file = os.path.join(opt.test_dataset_root, name + "." + opt.ground_truth_extension)
        coords = tb_data.parse_file(os.path.expanduser(gt_box_file))
        gt_coords = []
        for coord in coords:
            x1, x2 = min(coord[::2]), max(coord[::2])
            y1, y2 = min(coord[1::2]), max(coord[1::2])
            gt_coords.append([x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (70, 67, 238), 2)
        pred_final = []
        for box in bbox.bounding_boxes:
            x1, y1, x2, y2 = int(round(box.x1)), int(round(box.y1)), int(round(box.x2)), int(round(box.y2))
            pred_final.append([x1, y1, x2, y2])
            # box_tensors.append(torch.tensor([x1, y1, x2, y2]))
            # 4-point to 8-point: x1, y1, x2, y1, x2, y2, x1, y2
            f.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (x1, y1, x2, y1, x2, y2, x1, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 105, 65), 2)
        accu, precision, recall = measure(torch.Tensor(pred_final).cuda(), torch.Tensor(gt_coords).cuda(),
                                          width=img.shape[1], height=img.shape[0])
        precisions.append(precision)
        recalls.append(recall)
        img_save_directory = os.path.join(args.path, args.code_name, "val+" + "-".join(opt.model_prefix_list))
        if not os.path.exists(img_save_directory):
            os.mkdir(img_save_directory)
        name = "%s_%.2f_%.2f" % (str(i).zfill(4), precision, recall)
        cv2.imwrite(os.path.join(img_save_directory, name + ".jpg"), img)
        f.close()
        if opt.verbose:
            print("%d th image cost %.2f seconds, precision: %.2f, recall: %.2f" %
                  (i, time.time() - start, precision, recall))
    print("Precision: %.2f, Recall: %.2f" % (avg(precisions), avg(recalls)))
    # os.chdir(os.path.join(args.path, args.code_name, "result+"+ "-".join(opt.model_prefix_list)))
    # os.system("zip result_%s.zip ~/Pictures/dataset/ocr/_text_detection/result+%s/*.txt"
    # %("val+" + "-".join(opt.model_prefix_list), "-".join(opt.model_prefix_list)))


def avg(list):
    return sum(list) / len(list)


if __name__ == "__main__":
    opt = parse_arguments()
    with torch.no_grad():
        test_rotation(opt)