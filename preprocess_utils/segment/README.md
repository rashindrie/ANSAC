## Segmentation Model


Segmentation model weights are provided as in TF format.  Since I am primarily using Pytorch I convert the weights and model into Pytorch first.

This folder contains code necessary to:
1. [convert](./convert_tf_to_pytorch.py) the TF segmentation model weights to Pytorch weights and
2. [extract](./segment_wsi.py) segmentation information for vectorised WSIs.

```
python convert_tf_to_pytorch.py

mkdir pytorch_model
cd pytorch_model

python -m mmdnn.conversion._script.convertToIR \
    -f tensorflow \
    -d FCN8Model \
    -n ./tf_model/crowdsourcing_fcn8vgg16.ckpt.meta \
    --dstNodeName Squeeze \
    -w ./tf_model/crowdsourcing_fcn8vgg16.ckpt
    

python -m mmdnn.conversion._script.IRToCode \
    -f pytorch \
    --IRModelPath seg_fcn.pb \
    --dstModelPath seg_fcn.py \
    --IRWeightPath seg_fcn.npy \
    -dw seg_fcn_pytorch.npy
    
python -m mmdnn.conversion.examples.pytorch.imagenet_test \
    --dump seg_fcn.pth  \
    -n seg_fcn.py \
    -w seg_fcn_pytorch.npy
```

This will save the weights into `pytorch_model/seg_fcn_pytorch.npy` and also produce a file with a converted pytorch model `seg_fcn.py`.
However, I have made some further customizations on top of this file and provided the pytorch implementation for the [segmentation model here](segmentation_model.py).


