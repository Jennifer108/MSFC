
#==================================DEIT=====================================
# python tools/infer_camBefore-coco.py --model_path /data/fty/LZ/ToCo/results/toco_deit-b_coco_80k.pth \
#                                 --backbone deit_base_patch16_224  \
#                                 --data_folder  /data/fty/LZ/ToCo/MSCOCO/coco2014/ \
#                                 --list_folder  /data/fty/LZ/ToCo/datasets/coco/ \
#                                 --num_classes 81 \
#                                 --infer val \
#                                 --base_dir /data/fty/LZ/ToCo/COCO_VAL_DEIT_CAM

# without CRF
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tools/infer_seg_coco_ddp.py --model_path /data/fty/LZ/ToCo/results/toco_deit-b_coco_80k.pth --backbone deit_base_patch16_224 --infer val \
#         --base_dir  /data/fty/LZ/ToCo/COCO_VAL_DEIT_CAM


#=================================VIT============================================
# python tools/infer_camBefore-coco.py --model_path /data/fty/LZ/ToCo/results/toco_vit-b_coco_80k.pth \
#                                 --backbone vit_base_patch16_224  \
#                                 --data_folder  /data/fty/LZ/ToCo/MSCOCO/coco2014 \
#                                 --list_folder  /data/fty/LZ/ToCo/datasets/coco \
#                                 --num_classes 81 \
#                                 --infer val \
#                                 --base_dir /data/fty/LZ/ToCo/COCO_VAL_VIT_CAM


# with CRF
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tools/infer_seg_coco_ddp.py --model_path /home/newdisk/fty/LZ/ToCo/results/toco_vit-b_coco_80k.pth --backbone vit_base_patch16_224 --infer val \
        --base_dir  /home/newdisk/fty/LZ/ToCo/COCO_VAL_VIT_CAM

