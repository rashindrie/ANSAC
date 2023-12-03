import sys
sys.path.append('../')
import pdb
import torch
import numpy as np
from os.path import join
import torch.nn.functional as F

from clam.models.model_clam import CLAM_MB, CLAM_SB
from topk import SmoothTop1SVM
from core_utils.model import ANSAC
from core_utils.load_checkpoint import save_checkpoint, load_best_checkpoint


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feature_dirs = {
    'cleopatra': '',
}
weight_dirs = {
    'cleopatra': '',
}

extensions = {
    'cleopatra': '.ome.tif',
}

image_dirs = {
    'cleopatra': '',

}

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params



def get_input_features(image_name, dataset, ext='.ome.tif'):
    featurized_dir = feature_dirs[dataset]
    weight_dir = weight_dirs[dataset]

    full_image_name = image_name

    image_dirpath = f'{featurized_dir}/{full_image_name}/'
    weights_dirpath = f'{weight_dir}/{full_image_name}/'

    feature_filename = f'0_none_features.npy'
    weights_filename = f'0_none_features.npy'

    feature_path = join(image_dirpath, feature_filename)
    weights_path = join(weights_dirpath, weights_filename)

    features = np.load(feature_path)
    weights = np.load(weights_path)

    features = torch.from_numpy(features).float()
    weights = torch.from_numpy(weights).float()


    weights = weights.reshape(-1, 24, 24, 80, 80)
    weights = weights.permute(3, 4, 1, 2, 0)

    return features, weights


def infer_single_slide_clam(model, image_name, dataset, reverse_label_dict, label=0, k=1):
    images, _ = get_input_features(image_name, dataset)

    features = images.reshape(-1, 6400).transpose(1, 0)

    with torch.no_grad():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            features = features.to(device)
            logits, Y_prob, Y_hat, A, _ = model(features, return_features=True)
            Y_hat = Y_hat.item()

            if isinstance(model, (CLAM_MB,)):
                print("CLAM_MB")
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label,
                                                    ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

        del features

    return ids, preds_str, probs, A


def infer_single_slide_ansac(model, image_name, dataset, reverse_label_dict, label=0, k=1):
    features, weights = get_input_features(image_name, dataset)

    with torch.no_grad():
        logits, A, f_weighted, _ = model([features.unsqueeze(0) , weights.unsqueeze(0), device], output=True)

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = logits.argmax(dim=1, keepdim=True)
        Y_hat = Y_hat.item()

        A = A.view(-1, 1).cpu().numpy()

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A


def get_feature_extractor(model_type):
    from preprocess_utils.resnet_custom import resnet50_baseline
    from core_utils.load_checkpoint import get_moco_checkpoint
    if model_type == "CLAM":
        print('\ninitializing pretrained resnet as feature extractor model')
        feature_extractor = resnet50_baseline(pretrained=True)

        feature_extractor.eval()
        feature_extractor.to(device)
        print('Done!')
    elif model_type in ["CLAM_MOCO", 'ANSAC']:
        print('\ninitializing MoCo model as feature extractor from checkpoint')
        moco_configs = {
            'checkpoint_path': '/data/gpfs/projects/punim1193/wsi_classification_pipeline/final_models/moco/checkpoint.pth.tar',
            'arch': 'resnet50',
            'dim': 128,
            'k': 65536,
            'momentum': 0.999,
            'temp': 0.07,
            'mlp': False,
        }
        feature_extractor = get_moco_checkpoint(checkpoint_path=moco_configs['checkpoint_path'],
                                                arch=moco_configs['arch'],
                                                dim=moco_configs['dim'],
                                                k=moco_configs['k'],
                                                mom=moco_configs['momentum'],
                                                temp=moco_configs['temp'],
                                                mlp=moco_configs['mlp'])

        feature_extractor.eval()
        feature_extractor.to(device)
        print('Done!')
    else:
        feature_extractor = None
        print('Error: model type not recognized')

    return feature_extractor

def get_clam_model(ckpt_path, model_size='small'):
    label = 0

    print('\ninitializing model from checkpoint')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## CLAM
    model_dict = {"dropout": True, 'n_classes': 2, "size_arg": model_size, 'k_sample': 8}

    instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device)

    test_model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)

    test_model.load_state_dict(torch.load(ckpt_path))
    test_model.to(device)
    test_model.eval()

    print('\nckpt path: {}'.format(ckpt_path))
    return test_model


def get_ansac_model(ckpt_path, model_size=''):
    configs = {
        'batch_size': 1,
        'num_workers': 2,
        'model': {
            'size': [576, 256],
            'dropout': True,
            'group_norm': [1, 128],
            'normalize': True,
        },
    }

    print('\ninitializing ANSAC model from checkpoint')
    test_model = ANSAC(num_classes=2, **configs['model'])
    test_model = load_best_checkpoint(test_model, f'{ckpt_path}')

    test_model.to(device)
    test_model.eval()

    print('\nckpt path: {}'.format(ckpt_path))
    return test_model

