import torch
from os.path import isfile, join
from os import listdir
from PIL import Image
from pathlib import Path
import multiprocessing

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')

from core_utils.load_checkpoint import get_moco_checkpoint
from preprocess_utils.featurize_wsi import encode_augment_wsi_batch
from preprocess_utils.resnet_custom import resnet50_baseline

Image.MAX_IMAGE_PIXELS = 10000000000000


def multi_process_tasks(todo_tasks, function_to_run, num_process, k_args=[]):
    # how many processes to use
    # num_processes = multiprocessing.cpu_count()
    num_processes = num_process
    pool = multiprocessing.Pool(num_processes)

    feature_path = k_args[1]

    # comment this out
    # converted_list = [f for f in listdir(feature_path) if "processed.txt" in listdir(join(feature_path, f))]

    # converted_images = [f.split('.')[0] for f in converted_list]

    # final_list = []

    # for img in todo_tasks:
    #     if img.split('.')[0] not in converted_images:
    #         final_list.append(img)

    # todo_tasks = final_list
    # until here

    num_tasks = len(todo_tasks)

    if num_tasks > 0:
        if num_tasks != 0:
            if num_processes > num_tasks:
                num_processes = num_tasks
            images_per_process = num_tasks / num_processes

            print("Number of processes: " + str(num_processes))
            print("Number of training images: " + str(num_tasks))

            # each task specifies a range of slides
            tasks = []
            for num_process in range(1, num_processes + 1):

                start_index = (num_process - 1) * images_per_process
                end_index = num_process * images_per_process

                start_index = int(start_index)
                end_index = int(end_index)

                tasks.append(todo_tasks[start_index: end_index])

                if start_index == end_index:
                    print("Task #" + str(num_process) + ": Process slide " + str(start_index))
                else:
                    print(
                        "Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

            # start tasks
            results = []
            for t in tasks:
                results.append(pool.apply_async(function_to_run, args=[t, *k_args]))

            count = 0
            for result in results:
                count += result.get()
                if count == num_tasks:
                    print("\nDone processing %d tasks" % count)


def get_aug(train=False):
    if train:
        return [
            ('none', 0), ('none', 90), ('none', 180), ('none', 270),
            ('horizontal', 0), ('horizontal', 90), ('horizontal', 180), ('horizontal', 270),
            ('vertical', 0), ('vertical', 90), ('vertical', 180), ('vertical', 270),
            ('both', 0), ('both', 90), ('both', 180), ('both', 270),
            ]
    
    return [('none', 0)]


def pre_process_slides(slide_list, vector_path, feature_path, feature_encoder_model, code_size, train=False):
    
    for slide in slide_list:
        output_feature_directory = join(feature_path, slide)
        Path(output_feature_directory).mkdir(parents=True, exist_ok=True)

        processed_file_path = join(output_feature_directory, 'processed.txt')
        processed_slide_path = join(output_feature_directory, '270_both_features.npy')
        
        isfilepresent = isfile(processed_file_path)
        isslideprocessed = isfile(processed_slide_path)

        if not isfilepresent:
            f = open(processed_file_path, "w")
            f.close()

            wsi_pattern = join(vector_path, slide) + '/{item}.npy'

            try:
                encode_augment_wsi_batch(
                                 wsi_pattern=wsi_pattern, 
                                 encoder=feature_encoder_model.cuda(), 
                                 output_dir=output_feature_directory, 
                                 code_size=code_size, 
                                 batch_size=128, 
                                 aug_modes=get_aug(train), 
                                 save_features=True,
                                 overwrite=True)

                if isfile(processed_slide_path):
                    f = open(processed_file_path, "w")
                    f.write("Done")
                    f.close()
            except Exception as e: 
                print(f"**** EXCEPTION: {slide} ERROR: {e}")
                exit(0)

    return len(slide_list)


if __name__ == "__main__":
    vector_dirpath = sys.argv[1]
    feature_dirpath = sys.argv[2]
    model_name = sys.argv[3]
    model_ckpt_path = sys.argv[2]

    torch.multiprocessing.set_start_method('spawn')

    if model_name == 'moco':
        moco_configs = {
            'checkpoint_path': model_ckpt_path,
            'arch': 'resnet50',
            'dim': 128,
            'k': 65536,
            'momentum': 0.999,
            'temp': 0.07,
            'mlp': False,
        }
        encoder_model = get_moco_checkpoint(checkpoint_path=moco_configs['checkpoint_path'],
                                            arch=moco_configs['arch'],
                                            dim=moco_configs['dim'],
                                            k=moco_configs['k'],
                                            mom=moco_configs['momentum'],
                                            temp=moco_configs['temp'],
                                            mlp=moco_configs['mlp'])
        code_size = 128
    else:
        encoder_model = resnet50_baseline(pretrained=True)
        code_size = 1024

    save_features = True

    Path(feature_dirpath).mkdir(parents=True, exist_ok=True)

    all_images = [f for f in listdir(vector_dirpath) if f.endswith('.png')]

    multi_process_tasks(all_images, pre_process_slides, num_process=2, k_args=[vector_dirpath, feature_dirpath,
                                                                               encoder_model, code_size, True])
