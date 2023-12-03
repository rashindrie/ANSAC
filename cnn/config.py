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
            'group_norm': [1, 128],
        },
        'seeds': [42, 333, 2468, 1369, 2021, 21, 121, 8642, 7654, 2010],
        'weight_dir': None,
        'nic': True,
        'mixed_model': False,
    }
