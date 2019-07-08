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
