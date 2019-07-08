def GeneralPattern(args):
    args.path = "~/Pictures/dataset/ocr"
    # this will create a folder named "_text_detection" under "~/Pictures/dataset/ocr"
    args.code_name = "_text_detection"
    # Set it to True to make experiment result reproducible
    args.deterministic_train = False
    # Random seed for everything
    # If deterministic_train is disabled, then it will have no meaning
    args.seed = 1
    # Training Hyperparameter
    args.learning_rate = 1e-4
    args.batch_size_per_gpu = 1
    args.loading_threads = 1
    args.img_channel = 3
    args.epoch_num = 1000
    args.finetune = False

    # Because augmentation operation is defined in tb_augment.py
    args.do_imgaug = True

    # Image Normalization
    args.img_mean = (0.5, 0.5, 0.5)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args

def Unique_Patterns(args):
    args.train_sources = ["tempholding_auto_2"]
    args.train_aux = [{"txt": "txt", "img": "jpg"}]
    args.test_sources = ["tempholding"]
    args.test_aux = [{"txt": "xml", "img": "png"}]
    args.min_bbox_threshold = 0.01
    args.fix_size = True
    args.nms_threshold = 0.4
    args.augment_zoom_probability = 0.4
    args.augment_zoom_lower_bound = 1.3
    args.augment_zoom_higher_bound = 1.7

    args.cfg_super_wide = 0.5,
    args.cfg_super_wide_coeff = 0.5,
    args.model_prefix_finetune = "768",
    args.model_prefix = "768",
    args.jaccard_distance_threshold = 0.45,

    args.feature_pyramid_net = False
    args.self_attention = False
    args.focal_loss = False
    args.focal_power = 2.0

    args.inner_filters = 64
    args.inner_m_factor = 1
    return args


def Runtime_Patterns(args):
    args.train = False
    args.test = False
    args.with_extra4 = True

    args.loc_to_conf = True
    args.conf_incep = True
    args.loc_incep = True
    args.loc_preconv=True
    args.conf_preconv=True

    # Test Mode Only
    args.verbose = False
    args.nth_best_model = 1
    args.detector_top_k = 2500
    args.detector_conf_threshold = 0.05
    args.detector_nms_threshold = 0.3
    return args


PRESET = {
    "general": GeneralPattern,
    "unique": Unique_Patterns,
    "runtime": Runtime_Patterns,
}