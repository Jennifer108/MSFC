# 编译 Reg loss
# cd wrapper/bilateralfilter
# swig -python -c++ bilateralfilter.i   
# python setup.py install




# now
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  dist_train_voc_seg_negv2-Memorybank.py
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  dist_train_coco_seg_negv2-Memorybank.py










