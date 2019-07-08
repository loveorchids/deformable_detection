import os, glob, random, time
from collections import OrderedDict
import xml.etree.ElementTree as ET
import numpy as np
from os.path import *
import cv2

def remove_white_as_transparent(src, threshold=25):
    #src = cv2.imread(expanduser("~/Pictures/tmp.jpg"))
    tmp = cv2.bitwise_not(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY))
    _,alpha = cv2.threshold(tmp,threshold,255,cv2.THRESH_BINARY)
    return alpha.astype(float)/255
    #alpha = cv2.adaptiveThreshold(tmp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    #cv2.THRESH_BINARY,11,2)
    #b, g, r = cv2.split(src)
    #rgba = [b,g,r, alpha]
    #dst = cv2.merge(rgba,4)
    #cv2.imwrite(expanduser("~/Pictures/tmp.png"), dst)
    #return dst


def add_bg_object(key_info, invoice_canvas, alpha_paste, splitted_image_root):
    bg_img_list = glob.glob(join(splitted_image_root, "bg_noise", key_info[0], "*.jpg"))
    invoice_h, invoice_w = invoice_canvas.shape[0], invoice_canvas.shape[1]
    bg_num = random.randint(key_info[1][0], key_info[1][1])
    bg_img_list = random.sample(bg_img_list, bg_num)
    for bg_img in bg_img_list:
        bg_element = cv2.imread(bg_img)
        bg_h, bg_w = bg_element.shape[0], bg_element.shape[1]
        #bg_h = random.randint(round(bg_h * 0.8), round(invoice_h * 0.8))
        #bg_w = random.randint(round(bg_w * 0.8), round(invoice_w * 0.8))
        bg_h = random.randint(round(bg_h * 0.6), min(round(bg_h * 1.2), int(invoice_h * 0.9)))
        try:
            bg_w = random.randint(round(bg_w * 0.6), min(round(bg_w * 1.2), int(invoice_w * 0.9)))
        except ValueError:
            continue
        bg_element = cv2.resize(bg_element, (bg_w, bg_h))
        for r in range(random.randint(key_info[1][2], key_info[1][3])):
            start_x = random.randint(1, invoice_w - bg_w)
            start_y = random.randint(1, invoice_h - bg_h)
            if alpha_paste:
                alpha = remove_white_as_transparent(bg_element.astype("uint8"))
                bg_element = apply_alpha_matting(bg_element,
                                                 invoice_canvas[start_y: start_y + bg_h, start_x: start_x + bg_w, :3],
                                                 np.tile(np.expand_dims(alpha, -1), (1, 1, 3)))
            invoice_canvas[start_y: start_y + bg_h, start_x: start_x + bg_w, :3] = bg_element


def apply_alpha_matting(foreground, background, alpha):
    foreground = foreground.astype(float)
    background = background.astype(float)
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    return outImage

def auto_generate_data(splitted_image_root, output_dir, data_num=100, step_by_step=False):
    small_size = [18, 22]
    medium_size = [22, 26]
    large_size = [26, 34]
    super_large_size = [34, 46]

    all_splitted_img = glob.glob(splitted_image_root + "/*.jpg")

    invoice_size_mean = 0.7
    invoice_size_std = 0.1
    invoice_size_min = 0.5
    invoice_size_max = 0.7
    x_margin = 30
    y_margin = 10

    # Generate 500 invoice data
    total_added_img = 0
    alpha_paste = True
    bg_element_ref = OrderedDict({
        # first 2 dimensions: how many sample to get from folder
        # last 2 dimensions: how many times to repeatly add to the canvas
        "ceal": [0, 6, 1, 2],
        "table": [0, 2, 1, 1],
        "pattern": [2, 4, 2, 4],
        "mark": [2, 4, 3, 9],
    })

    # Generating Process
    for i in range(data_num):
        start_time = time.time()
        invoice_long_side = random.randint(1536, 2048)
        name = "img_%s"%(str(i).zfill(4))
        short_side = int(min(max(random.gauss(invoice_size_mean, invoice_size_std),
                                 invoice_size_min), invoice_size_max) * invoice_long_side)
        if random.uniform(0, 1) > 1:
            invoice_canvas = np.ones((short_side, invoice_long_side, 3)).astype(np.uint8) * 255
            invoice_usage = np.zeros((short_side, invoice_long_side)).astype(np.uint8)
        else:
            # height = 2048 and height > width
            invoice_canvas = np.ones((invoice_long_side, short_side, 3)).astype(np.uint8) * 255
            invoice_usage = np.zeros((invoice_long_side, short_side)).astype(np.uint8)
        invoice_h, invoice_w = invoice_canvas.shape[0], invoice_canvas.shape[1]
        img_in_use = random.sample(all_splitted_img, round(random.gauss(50, 1)))
        usage_sum = 0

        # Add background element
        for key_pair in bg_element_ref.items():
            add_bg_object(key_pair, invoice_canvas, alpha_paste, splitted_image_root)

        # Add image element
        f = open(join(output_dir, name + ".txt"), "w")
        added_img_num = 0
        for j, img_file in enumerate(img_in_use):
            img = cv2.imread(img_file)
            if img is None:
                continue
            if len(img.shape) == 2:
                img = np.tile(np.expand_dims(img, -1), (1, 1, 3))
            h, w = img.shape[0], img.shape[1]
            ratio = w / h
            if ratio < 1:
                size_indicator = random.uniform(0, 0.4)
                repeat = random.randint(5, 15)
            elif 1 <= ratio < 2:
                size_indicator = random.uniform(0, 0.5)
                repeat = random.randint(1, 10)
            elif 2 <= ratio < 5:
                size_indicator = random.gauss(0.75, 0.5)
                repeat = 1
            elif 5 <= ratio < 10:
                size_indicator = random.uniform(0, 0.5)
                repeat = random.randint(1, 5)
            else:
                size_indicator = random.uniform(0, 0.4)
                repeat = random.randint(1, 5)
            for r in range(repeat):
                if size_indicator < 0.3:
                    height = round(random.uniform(small_size[0], small_size[1]))
                elif size_indicator < 0.6:
                    height = round(random.uniform(medium_size[0], medium_size[1]))
                elif size_indicator < 0.9:
                    height = round(random.uniform(large_size[0], large_size[1]))
                else:
                    height = round(random.uniform(super_large_size[0], super_large_size[1]))
                # ratio preserved resize
                if round(w * height / h) == 0:
                    continue
                img = cv2.resize(img, (round(w * height / h), height))
                h, w = img.shape[0], img.shape[1]
                if alpha_paste:
                    # add alpha channel
                    alpha = remove_white_as_transparent(img)
                # Decide the x1, y1
                failure_time = 0
                while failure_time < 12:
                    if invoice_w - w -1 > 0:
                        x1 = random.randint(0, invoice_w - w -1)
                    else:
                        x1 = 0
                    if invoice_h - h - 1 > 0:
                        y1 = random.randint(0, invoice_h - h - 1)
                    else:
                        y1 = 0
                    if x1 + w > invoice_w:
                        x_max = invoice_w
                    else:
                        x_max = x1 + w
                    if y1 + h > invoice_h:
                        y_max = invoice_h
                    else:
                        y_max = y1 + h
                    # See if invoice_usage is almost not used
                    if np.sum(invoice_usage[
                              max(y1-y_margin, 0): min(y_max+y_margin, invoice_h),
                              max(x1-x_margin, 0): min(x_max+x_margin, invoice_w)]
                              ) < 5:
                        alpha_img = apply_alpha_matting(
                            foreground=img[: y_max, : x_max, :],
                            background=invoice_canvas[y1: y_max, x1: x_max, :],
                            alpha=np.tile(np.expand_dims(alpha[: y_max, : x_max], -1), (1, 1, 3))
                            #alpha=alpha[: y_max, : x_max]
                        )
                        invoice_canvas[y1: y_max, x1: x_max, :] = alpha_img
                        invoice_usage[max(y1-y_margin, 0): min(y_max+y_margin, invoice_h),
                              max(x1-x_margin, 0): min(x_max+x_margin, invoice_w)] = 1
                        if np.sum(invoice_usage) > usage_sum:
                            usage_sum = np.sum(invoice_usage)
                        else:
                            print("Error")
                        x2, y2 = x_max, y_max
                        f.write("%d,%d,%d,%d,%d,%d,%d,%d, \n" % (x1, y1, x2, y1, x2, y2, x1, y2))
                        added_img_num += 1
                        if step_by_step:
                            cv2.imwrite(join(output_dir, "tmp_%d_%d.jpg"%(j, r)), invoice_canvas)
                        break
                    else:
                        failure_time += 1
        f.close()
        f = open(join(output_dir, name + ".txt"), "r").readlines()
        assert len(f) == added_img_num
        cv2.imwrite(join(output_dir, name + ".jpg"), invoice_canvas)
        total_added_img += added_img_num
        print("%d-th Canvas shape: (%d, %d) has %s img added, cost %.2f seconds" %
              (i, invoice_canvas.shape[0], invoice_canvas.shape[1], str(added_img_num).zfill(3), time.time() - start_time))
    print("Totally this dataset contains %d annotated boxes" %(total_added_img))


def create_dataset(root_path, new_dataset_path, img_operation=True):
    char_statistics = {}
    word_set = set([])
    #f = open(os.path.join(new_dataset_path, "label.txt"), "w")
    txt_list = sorted(glob.glob(root_path + "/*.xml"))
    img_num = 0
    for i, txt_file in enumerate(txt_list):
        start = time.time()
        name = txt_file[txt_file.rfind("/") + 1 : -4]
        img_path = os.path.join(root_path, name + ".png")
        if not os.path.exists(img_path):
            print("image: %s does not exists"%(name + ".png"))
            continue
        if img_operation:
            img = cv2.imread(img_path)
        else:
            img = None
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
        for coord in coords:
            x1, x2 = min(coord[::2]), max(coord[::2])
            y1, y2 = min(coord[1::2]), max(coord[1::2])
            if img_operation:
                img_with_txt = img[y1: y2, x1: x2, :]
                cv2.imwrite(os.path.join(new_dataset_path, str(img_num) + ".jpg"), img_with_txt)
                #write_line = str(img_num) + ".jpg:%s"%(text_label)
                #f.write(write_line)
            img_num += 1
        print("%d/%d completed, cost %.2f seconds"%(i, len(txt_list), time.time() - start))
    #f.close()
    return word_set, char_statistics


if __name__ == "__main__":
    root_path = os.path.expanduser("~/Pictures/dataset/ocr/tempholding")
    new_dataset_path = os.path.expanduser("~/Pictures/dataset/ocr/tempholding_crop")
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
    #word_set, char_statistics = create_dataset(root_path, new_dataset_path, img_operation=True)
    output_dir = expanduser("~/Pictures/dataset/ocr/tempholding_auto")
    #output_dir = expanduser("~/Pictures")
    auto_generate_data(new_dataset_path, output_dir, data_num=500, step_by_step=False)