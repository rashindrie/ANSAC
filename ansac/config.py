configurations = {
        'seed': 42,  # random seed for reproducible experiment (default: 42)
        'batch_size': 16,
        'num_workers': 8,
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0,
        'scheduler_step': [120, 180],
        'start_epoch': 0,
        'epochs': 300,
        'resume': None,
        'print_freq': 10,
        'test_freq': 50,
        'early_stop_count': 25,
        'num_repeats': 1,
        'model': {
            'size': [576, 256],
            'dropout': True,
            'group_norm': [1, 128],
            'normalize': True,
        },
        'seeds': [42, 333, 2468, 1369, 2021, 21, 121, 8642, 7654, 2010],
        'nic': False,
        'mixed_model': False,
    }


def get_configs(argv):
    input_csv_dir = argv[2]
    input_data_dir = argv[3]
    input_weight_dir = argv[4]
    input_ext = argv[5]

    configurations['feature_dir'] = input_data_dir
    configurations['weight_dir'] = input_weight_dir
    configurations['csv_dir'] = input_csv_dir
    configurations['ext'] = input_ext

    try:
        input_ckpt_dir = argv[6]
        configurations['ckpt_dir'] = input_ckpt_dir
    except:
        pass

    if input_data_dir == "None":
        configurations['feature_dir'] = {
            'cleopatra': '/data/Cleopatra/data/featurized_wsi/',
            'marianne': '/data/MARIANNE/data/featurized_wsi/',
            'finher': '/data/FinHer/data/featurized_wsi/',
            'tcga': '/data/TCGA/data/featurized_wsi/'
        }
        configurations['weight_dir'] = {
            'cleopatra': '/data/Cleopatra/data/segmented_wsi/',
            'marianne': '/data/MARIANNE/data/segmented_wsi/',
            'finher': '/data/FinHer/data/segmented_wsi/',
            'tcga': '/data/TCGA/data/segmented_wsi/'
        }
        configurations['mixed_model'] = True
    return configurations
