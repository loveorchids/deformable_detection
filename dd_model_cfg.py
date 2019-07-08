

ssd_512 = {
    # Configuration for 512x512 input image
    'num_classes': 2,
    'conv_output': ["conv_3", "conv_5", "extra_1", "extra_2", "extra_3", "extra_4"],
    #'feature_map_sizes': [96, 48, 24, 24, 24],
    'feature_map_sizes': [64, 16, 16, 8, 4, 2],
    'input_img_size': [512, 512],
    #'input_img_size': [768, 768],
    # See the visualization result by enabling visualize_bbox in function fit of textbox.py
    # And change the settings according to the result
    # Some possible settings of box_height and box_height_large
    # 'box_height': [[16], [26], [36]],
    # 'box_height': [[10, 16], [26], [36]],
    # 'box_height': [[16], [26], []],
    'box_height': [[18], [30], [46], [68]],
    #'box_ratios': [[1, 2, 4, 7, 11, 15, 20, 26], [0.5, 1, 2, 5, 9, 13, 16, 18],
                   #[0.25, 0.5, 1, 2, 5, 8, 10], [0.5, 1, 2, 3, 5, 8]],
    'box_ratios': [[0.333, 0.25, 1, 2, 3], [0.333, 0.25, 1, 2, 3],
                   [0.333, 0.25, 1, 2, 3], [0.333, 0.25, 1, 2, 3], [0.333, 0.25, 1, 2, 3]],
    # If big_box is True, then box_height_large and box_ratios_large will be used
    'big_box': True,
    'box_height_large': [[24], [38], [56], [98]],
    #'box_ratios_large': [[0.5, 1, 2, 4, 7, 11, 15, 20], [0.3, 0.5, 1, 3, 6, 9, 11, 13],
                         #[0.25, 0.5, 1, 2, 4, 7, 9], [0.5, 1, 2, 3, 5]],
    'box_ratios_large': [[1, 2, 4, 7, 11, 15, 20], [0.5, 1, 3, 6, 9, 13],
                         [1, 2, 4, 7, 9], [1, 2, 3, 5], [1, 2, 3, 4]],
    # You can increase the stride when feature_map_size is large
    # especially at swallow conv layers, so as not to create lots of prior boxes
    'stride': [1, 1, 1, 1],
    # Input depth for location and confidence layers
    'loc_and_conf': [256, 512, 384, 256],
    # The hyperparameter to decide the Loss
    'variance': [0.1, 0.2],
    'var_updater': 1,
    'alpha': 1,
    'alpha_updater': 1,
    # Jaccard Distance Threshold
    'overlap_thresh': 0.45,
    # Whether to constrain the prior boxes inside the image
    'clip': True,
    'super_wide': 0.5,
    'super_wide_coeff': 0.5,
}