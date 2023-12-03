import sys

import os
import torch
import torchvision.models as models
from collections import OrderedDict

sys.path.append('../')
sys.path.append('../../')

from moco import builder


def convert_state_dict(state_dict, text="module."):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith(text):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[len(text):]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_moco_checkpoint(checkpoint_path, arch, dim, k, mom, temp, mlp):
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    print("=> creating model '{}'".format(arch))
    model = builder.MoCo(models.__dict__[arch], dim, k, mom, temp, mlp)

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(convert_state_dict(checkpoint['state_dict']))
        print("=> loaded checkpoint '{}'".format(checkpoint_path))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return model.encoder_q


def get_checkpoint_for_transfer_learning(checkpoint_path, arch, dim=128):
    # create model
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    # model.fc = nn.Linear(2048, dim)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    print("=> loaded pre-trained model '{}'".format(checkpoint_path))
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        # shutil.copyfile(filename, 'model_best.pth.tar')


def load_best_checkpoint(model, checkpoint_file):
    if os.path.isfile(checkpoint_file):
        # print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(convert_state_dict(checkpoint['state_dict']))
        # print("=> loaded checkpoint '{}' (epoch {})"
        #       .format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))
        return None
    return model

