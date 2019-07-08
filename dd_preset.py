import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Textbox Detector Settings')
    ##############
    #        TRAINING        #
    ##############
    parser.add_argument(
        "--train",
        action="store_true",
        help="enable train",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="enable test",
    )
    parser.add_argument(
        "-dt",
        "--deterministic_train",
        action="store_true",
        help="if this is turned on, then everything will be deterministic"
             "and the training process will be reproducible.",
    )
    parser.add_argument(
        "-cdb",
        "--cudnn_benchmark",
        action="store_true",
        help="Turn on CuDNN benchmark",
    )
    parser.add_argument(
        "-ft",
        "--finetune",
        action="store_true",
        help="do finetune",
    )
    parser.add_argument(
        "-en",
        "--epoch_num",
        type=int,
        help="Epoch number of the training",
        default=200
    )

    parser.add_argument(
        "-mp",
        "--model_prefix",
        type=str,
        help="prefix of model",
    )
    parser.add_argument(
        "-mpf",
        "--model_prefix_finetune",
        type=str,
        help="prefix of existing model need to be finetuned",
    )
    parser.add_argument(
        "-bpg",
        "--batch_size_per_gpu",
        type=int,
        help="batch size inside each GPU during training",
        default=6
    )
    parser.add_argument(
        "-lt",
        "--loading_threads",
        type=int,
        help="loading_threads correspond to each GPU during both training and validation, "
             "e.g. You have 4 GPU and set -lt 2, so 8 threads will be used to load data",
        default=2
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs='+',
        help="a list folder/folders to use as training set",
        default=["tempholding_auto"]
    )

    ##############
    #   AUGMENTATION   #
    ##############
    parser.add_argument(
        "-azp",
        "--augment_zoom_probability",
        type=float,
        help="Probability of zoom the input image",
        default=0.5
    )
    parser.add_argument(
        "-azlb",
        "--augment_zoom_lower_bound",
        type=float,
        help="lower bound of zoom rate",
        default=1.2
    )
    parser.add_argument(
        "-azhb",
        "--augment_zoom_higher_bound",
        type=float,
        help="higher bound of zoom rate",
        default=1.6
    )

    ##############
    #          MODEL         #
    ##############
    parser.add_argument(
        "--with_extra4",
        type=bool,
        default=True,
        help="with extra 4 layer in model defination",
    )
    parser.add_argument(
        "-csw",
        "--cfg_super_wide",
        type=float,
        help="probability of using super wide box matching mechanism",
        default=0.3
    )
    parser.add_argument(
        "-cswc",
        "--cfg_super_wide_coeff",
        type=float,
        help="to suppress or increase the matched super wide overlap"
             "<1 means suppress, >1 means increase",
        default=0.5
    )
    parser.add_argument(
        "-jdt",
        "--jaccard_distance_threshold",
        type=float,
        help="The jaccard distance of prior box and target below this value will"
             "be treated as negative value.",
        default=0.45
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
        "-fl",
        "--focal_loss",
        action="store_true",
        help="use focal_loss",
    )
    parser.add_argument(
        "-fp",
        "--focal_power",
        type=float,
        default=2.0,
        help="power of focal loss",
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
    ##############
    #          TEST         #
    ##############
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="output test score for each sample",
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
    args = parser.parse_args()
    return args


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