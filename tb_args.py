import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Textbox Detector Settings')
    ##############
    #        TRAINING        #
    ##############
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
        required=True
    )
    parser.add_argument(
        "-mpf",
        "--model_prefix_finetune",
        type=str,
        help="prefix of existing model need to be finetuned",
        required=True
    )
    parser.add_argument(
        "-bpg",
        "--batch_size_per_gpu",
        type=int,
        help="batch size inside each GPU during training",
        default=1
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
        default=["tempholding"]
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
    
    

    args = parser.parse_args()
    return args
