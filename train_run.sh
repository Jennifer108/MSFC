# Reg loss
# cd wrapper/bilateralfilter
# swig -python -c++ bilateralfilter.i   
# python setup.py install



CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  train_voc_seg.py
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  train_coco_seg.py










