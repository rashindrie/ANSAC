import os
import sys
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,64).__str__()

import numpy as np
import argparse, yaml
import matplotlib.pyplot as plt
import torch.utils.data.distributed

from wsi_core.wsi_utils import sample_rois
from wsi_core.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches, score2percentile
from heatmap_generation.heatmap_custom_utils import infer_single_slide_clam, parse_config_dict, \
    get_feature_extractor, get_clam_model, infer_single_slide_ansac, get_ansac_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def run_for_image(slide_name, thresh_, label, config_dict):
    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    print("Setting threshold to: ",thresh_)
    heatmap_args.binary_thresh = thresh_

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}


    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
        'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
    label_dict =  data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

    slide_id = slide_name.replace(data_args.slide_ext, '')

    p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code,  slide_id, model_type)
    os.makedirs(p_slide_save_dir, exist_ok=True)

    r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code,  slide_id, model_type)
    os.makedirs(r_slide_save_dir, exist_ok=True)

    top_left = None
    bot_right = None

    print('slide id: ', slide_id)
    print('top left: ', top_left, ' bot right: ', bot_right)

    if isinstance(data_args.data_dir, str):
        slide_path = os.path.join(data_args.data_dir, slide_name)
    # elif isinstance(data_args.data_dir, dict):
    #     data_dir_key = process_stack.loc[i, data_args.data_dir_key]
    #     slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
    else:
        raise NotImplementedError
    mask_file = os.path.join(r_slide_save_dir, 'mask.pkl')

    # Load segmentation and filter parameters
    seg_params = def_seg_params.copy()
    filter_params = def_filter_params.copy()
    vis_params = def_vis_params.copy()

    # seg_params = load_params(process_stack.loc[i], seg_params)
    # filter_params = load_params(process_stack.loc[i], filter_params)
    # vis_params = load_params(process_stack.loc[i], vis_params)

    keep_ids = str(seg_params['keep_ids'])
    if len(keep_ids) > 0 and keep_ids != 'none':
        seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
    else:
        seg_params['keep_ids'] = []

    exclude_ids = str(seg_params['exclude_ids'])
    if len(exclude_ids) > 0 and exclude_ids != 'none':
        seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
    else:
        seg_params['exclude_ids'] = []

    for key, val in seg_params.items():
        print('{}: {}'.format(key, val))

    for key, val in filter_params.items():
        print('{}: {}'.format(key, val))

    for key, val in vis_params.items():
        print('{}: {}'.format(key, val))

    print('Initializing WSI object')
    wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
    print('Done!')

    wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

    # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
    vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

    block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
    mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
    if vis_params['vis_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        vis_params['vis_level'] = best_level
    mask = wsi_object.visWSI(**vis_params, number_contours=True)
    mask.resize(heatmap_size).save(mask_path)
    wsi_object.saveSegmentation(mask_file)


    if model_type in ['CLAM', 'CLAM_MOCO']:
        partial_fn = infer_single_slide_clam
    else:
        partial_fn = infer_single_slide_ansac

    Y_hats, Y_hats_str, Y_probs, A = partial_fn(test_model, slide_name, 'cleopatra',
                                                             reverse_label_dict,
                                                             label=label, k=exp_args.n_classes)

    scores = A[:,0]
    coords = [(x*256,y*256) for x in range(80) for y in range(80)] 
    coords = np.array(coords)


    samples = sample_args.samples
    for sample in samples:
        if sample['sample']:
            # tag = "slide_name_{}_label_{}_pred_{}".format(slide_id, label, Y_hats[0])
            sample_save_dir =  os.path.join(r_slide_save_dir, 'sampled_patches', sample['name'])
            os.makedirs(sample_save_dir, exist_ok=True)
            print('sampling {}'.format(sample['name']))
            sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
            for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))


    wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
    'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

    heatmap_save_name = '{}_{}_blockmap.tiff'.format(slide_id, heatmap_args.binary_thresh)
    if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
        pass
    else:
        heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, 
                              cmap=heatmap_args.cmap, alpha=heatmap_args.alpha,
                              use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                              thresh=heatmap_args.binary_thresh, patch_size = vis_patch_size, 
                              convert_to_percentiles=True
                             )
        heatmap = heatmap.resize(heatmap_size)
        heatmap.save(os.path.join(r_slide_save_dir, '{}_{}_blockmap.png'.format(slide_id, heatmap_args.binary_thresh)))
        print(f"Heatmap saved in : {r_slide_save_dir}")
        plt.imshow(heatmap.resize((224, 224)))
        plt.show()
        del heatmap



    if heatmap_args.use_ref_scores:
        ref_scores = scores.copy()
    else:
        ref_scores = None

    # if heatmap_args.calc_heatmap:
    #     compute_from_patches_with_segmentation(wsi_object=wsi_object, clam_pred=Y_hats[0], model=test_model, 
    #                                            feature_extractor=feature_extractor, segmentor=segmentor,
    #                          batch_size=exp_args.batch_size, **wsi_kwargs, 
    #                         attn_save_path=save_path,  ref_scores=ref_scores)

    # if not os.path.isfile(save_path):
    #     print('heatmap {} not found'.format(save_path))
    #     if heatmap_args.use_roi:
    #         save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
    #         print('found heatmap for whole slide')
    #         save_path = save_path_full
    #     else:
    #         pass


    # file = h5py.File(save_path, 'r')
    # dset = file['attention_scores']
    # coord_dset = file['coords']
    # scores = dset[:]
    # coords = coord_dset[:]
    # file.close()

    A2 = A.copy()
    A2.shape, A.shape, np.max(A2)


    for score_idx in range(len(A2)):
        A2[score_idx] = score2percentile(A2[score_idx], ref_scores)

    heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level,
                        'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
    if heatmap_args.use_ref_scores:
        heatmap_vis_args['convert_to_percentiles'] = False

    heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
                                                                                    int(heatmap_args.blur), 
                                                                                    int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
                                                                                    float(heatmap_args.alpha), int(heatmap_args.vis_level), 
                                                                                    int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


    if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
        print("Doing nothing...")
        pass

    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        heatmap = drawHeatmap(A2, coords, slide_path, wsi_object=wsi_object,  
                              cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
                              binarize=heatmap_args.binarize, 
                              blank_canvas=heatmap_args.blank_canvas,
                              thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
                              overlap=patch_args.overlap, 
                              top_left=top_left, bot_right = bot_right)

        heatmap = heatmap.resize(heatmap_size)
        if heatmap_args.save_ext == 'jpg':
            heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
        else:
            heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

        plt.imshow(heatmap.resize((224, 224)))
        plt.show()
        
        del heatmap
        
    del wsi_object, A


if __name__ == '__main__':
    slide_id = sys.argv[1]
    print_args = True
    heatmap_size = (2000, 2000)

    for thresh_ in [-1, 0.2, 0.4, 0.6, 0.8]:
        parser = argparse.ArgumentParser(description='Heatmap inference script')
        parser.add_argument('--save_exp_code', type=str, default=None, help='experiment code')
        parser.add_argument('--overlap', type=float, default=None)
        parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
        args = parser.parse_args([])

        config_path = os.path.join('./configs.yaml')
        config_dict = yaml.safe_load(open(config_path, 'r'))
        config_dict = parse_config_dict(args, config_dict)
        model_type = config_dict.model_arguments.model_type.upper()
        ckpt_path = config_dict.model_arguments.ckpt_path

        if print_args:
            # we load the models and print the default args only once
            print("Generating heatmaps for model type: ", model_type)

            if model_type == 'ANSAC':
                partial_fn = get_ansac_model
            else:
                partial_fn = get_clam_model

            test_model = partial_fn(ckpt_path=ckpt_path, model_size=config_dict.model_arguments.model_size)
            feature_extractor = get_feature_extractor(model_type)

            for key, value in config_dict.items():
                if isinstance(value, dict):
                    print('\n' + key)
                    for value_key, value_value in value.items():
                        print(value_key + " : " + str(value_value))
                else:
                    print('\n' + key + " : " + str(value))
            print_args = False

        print("Starting to process: ", slide_id, thresh_)
        run_for_image(slide_name = slide_id, thresh_=thresh_, label=0, config_dict=config_dict)

    del feature_extractor, test_model
