import os, glob
import torch, cv2, imgaug
import xml.etree.ElementTree as ET
from torch.utils.data import *
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset
import omni_torch.data.data_loader as omth_loader
import omni_torch.utils as util
from researches.ocr.textbox.tb_preprocess import *
from researches.ocr.textbox.tb_augment import *


def get_path_and_label(args, length, paths, auxiliary_info):
    img_files, txt_files = [], []
    path_list = sorted(glob.glob(paths+ "/*.%s"%(auxiliary_info["txt"])))
    for i, txt_file in enumerate(path_list):
        img_name = txt_file[txt_file.rfind("/") + 1:-4]
        img_path = os.path.join(paths, img_name + ".%s" % (auxiliary_info["img"]))
        if not os.path.exists(img_path):
            continue
        img_files.append(img_path)
        txt_files.append(txt_file)
    return [list(zip(img_files, txt_files))]


def extract_bbox(args, path, seed, size):
    img_file, txt_file = path[0], path[1]
    if args.img_channel is 1:
        image = cv2.imread(img_file, 0)
    else:
        image = cv2.imread(img_file)
    h, w = image.shape[0], image.shape[1]
    coords = parse_file(txt_file)
    BBox=[]
    for coord in coords:
        x1, x2 = min(coord[::2]), max(coord[::2])
        y1, y2 = min(coord[1::2]), max(coord[1::2])
        if abs(x2 - x1) * abs(y2 - y1) <= args.min_bbox_threshold * h * w / 100:
            # Skip a bbox which is smaller than a certain percentage of the total size
            continue
        BBox.append(imgaug.augmentables.bbs.BoundingBox(x1, y1, x2, y2))
    BBox = imgaug.augmentables.bbs.BoundingBoxesOnImage(BBox, shape=image.shape)
    # The one with text is labeled as 0 not 1, or that would cause trouble in loss calculations
    return image, BBox, [0 for i in coords]


def parse_file(txt_file):
    coords = []
    if txt_file.endswith("txt"):
        with open(txt_file, mode="r") as txtfile:
            for line in txtfile:
                coord = line.strip().split(",")[:8]
                coord = [int(c) for c in coord]
                coords.append(coord)
    elif txt_file.endswith("xml"):
        prefix = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15}'
        tree = ET.parse(txt_file)
        root = tree.getroot()
        coords = []
        for node in root[1].findall("%sTextRegion" % (prefix)):
            for n in node.findall("%sTextLine" % (prefix)):
                for coord in n.findall("%sCoords" % (prefix)):
                    co = []
                    points = coord.items()[0][1].strip().split()
                    for point in points:
                        co += [int(p) for p in point.split(",")]
                    coords.append(co)
    else:
        raise NotImplementedError
    return coords


def detection_collector(batch):
    imgs, labels = [], []
    for sample in batch:
        if sample[0][1].size(0) == 0 or sample[0][2].size(0) == 0:
            # There is no bbox or label inside the image
            continue
        imgs.append(sample[0][0])
        labels.append(torch.cat([sample[0][1], sample[0][2].unsqueeze(-1)], dim=1))
    try:
        imgs = torch.stack(imgs, 0)
    except:
        imgs = batch[0][0][0].unsqueeze(0)
    return imgs, labels


def fetch_detection_data(args, sources, auxiliary_info, batch_size, batch_size_val=None,
                         shuffle=True, split_val=0.0, k_fold=1, pre_process=None, aug=None):
    args.loading_threads = round(args.loading_threads * torch.cuda.device_count())
    batch_size = round(batch_size * torch.cuda.device_count())
    if batch_size_val is None:
        batch_size_val = batch_size
    else:
        batch_size_val * torch.cuda.device_count()
    dataset = []
    for i, source in enumerate(sources):
        subset = Arbitrary_Dataset(args, sources=[source], step_1=[get_path_and_label],
                                   step_2=[omth_loader.read_image_with_bbox], bbox_loader=[extract_bbox],
                                   auxiliary_info=[auxiliary_info[i]], pre_process=[pre_process],
                                   augmentation=[aug])
        subset.prepare()
        dataset.append(subset)

    if k_fold > 1:
        return util.k_fold_cross_validation(args, dataset, batch_size, batch_size_val,
                                            k_fold, collate_fn=detection_collector)
    else:
        if split_val > 0:
            return util.split_train_val_dataset(args, dataset, batch_size, batch_size_val,
                                                split_val, collate_fn=detection_collector)
        else:
            kwargs = {'num_workers': args.loading_threads, 'pin_memory': True}
            train_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                                   shuffle=shuffle, collate_fn=detection_collector, **kwargs)
            return [(train_set, None)]


if __name__ == "__main__":
    pass
