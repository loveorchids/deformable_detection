from imgaug import augmenters

def aug_temp(args, bg_color=255):
    aug_list = []
    stage_0, stage_1, stage_2, stage_3 = 2048, 2048, 512, 512

    # Pad the height to stage_0
    aug_list.append(augmenters.PadToFixedSize(width=1, height=stage_0, pad_cval=bg_color))
    # Resize its height to stage_1, note that stage_0 is smaller than stage_1
    # so that the font size could be increased for most cases.
    aug_list.append(augmenters.Resize(size={"height": stage_1, "width": "keep-aspect-ratio"}))
    # increase the aspect ratio
    aug_list.append(augmenters.Sometimes(args.augment_zoom_probability,
        augmenters.Affine(scale=(args.augment_zoom_lower_bound, args.augment_zoom_higher_bound))
    ))

    # Crop a stage_2 x stage_2 area
    aug_list.append(augmenters.CropToFixedSize(width=stage_2, height=stage_2))
    # In case the width is not enough, pad it to stage_2 x stage_2
    aug_list.append(augmenters.PadToFixedSize(width=stage_2, height=stage_2, pad_cval=bg_color))

    # Resize to stage_3 x stage_3
    #aug_list.append(augmenters.Resize(size={"height": stage_3, "width": stage_3}))

    # Perform Flip
    aug_list.append(augmenters.Fliplr(0.33, name="horizontal_flip"))
    #aug_list.append(augmenters.Flipud(0.33, name="vertical_flip"))

    # Perform Contrast Augmentation
    aug_list.append(augmenters.Sometimes(0.5, augmenters.GammaContrast(gamma=(0.5, 1.2))))
    aug_list.append(augmenters.Sometimes(0.5, augmenters.LinearContrast(alpha=(0.4, 1.2))))
    return aug_list