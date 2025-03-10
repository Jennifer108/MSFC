

#==================================DEIT=====================================
# python tools/infer_camBefore.py --model_path /data/fty/LZ/MSFC/results/MSFC_deit-b_voc_20k.pth --backbone deit_base_patch16_224  \
#             --infer val --base_dir /data/fty/LZ/MSFC/VOC_VAL_DEIT_CAM


# without CRF
# python tools/infer_seg_voc.py --model_path /data/fty/LZ/MSFC/results/MSFC_deit-b_voc_20k.pth --backbone deit_base_patch16_224 --infer val \
#         --base_dir  /data/fty/LZ/MSFC/VOC_VAL_DEIT_CAM



#=================================VIT============================================
# python tools/infer_camBefore.py --model_path /data/fty/LZ/MSFC/results/MSFC_vit-b_voc_20k.pth --backbone vit_base_patch16_224 \
#          --infer val --base_dir /data/fty/LZ/MSFC/VOC_VAL_VIT_CAM


# with CRF
python tools/infer_seg_voc.py --model_path /data/fty/LZ/MSFC/results/MSFC_vit-b_voc_20k.pth --backbone vit_base_patch16_224 --infer val \
    --base_dir  /data/fty/LZ/MSFC/VOC_VAL_VIT_CAM

