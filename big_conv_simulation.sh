#####################################################################
#####################################################################
####################### hyperparameter search #######################
#####################################################################
#####################################################################
#psf_path="data/simulation/opt_radial_mask.npy"
#gt_path="/media/vieira/03308fae-a4c4-4200-b84c-199e55907e21/datasets/FFHQ/images1024x1024/00000/00000.png"
#gt_reso=256
#for tv_alpha in 0.0000003 0.0000005 0.0000007 # 0.000003 0.000005 0.000007 0.000009 # 0.000000001 0.00000001 0.0000001 0.000001 0.00001
#do
#  python big_conv.py --gt_path $gt_path --gt_reso $gt_reso --sigma 0.005 \
#  --psf_path $psf_path --psf_reso 820 --iters 10000 --mask_type "opt_radial"\
#  --tv_alpha $tv_alpha  --save_path "results/big_conv/param_search/opt_radial/${tv_alpha}/"
#
#  python big_conv.py --gt_path $gt_path --gt_reso $gt_reso --sigma 0.005 \
#  --psf_path $psf_path --psf_reso 512 --psf_center_crop_size 128 --iters 10000 --mask_type "restricted" \
#  --tv_alpha $tv_alpha  --save_path "results/big_conv/param_search/restricted6per/${tv_alpha}/"
#done

#####################################################################
#####################################################################
####################### Quantitative analysis #######################
#####################################################################
#####################################################################
psf_path="data/simulation/opt_radial_mask.npy"
gt_base_path="/media/vieira/03308fae-a4c4-4200-b84c-199e55907e21/datasets/FFHQ/images1024x1024/00000/" # Path to FFHQ dataset
gt_reso=256
tv_alpha_full=0.0000001
tv_alpha_rest=0.000003
for i in {0..100}
do
  printf -v img_num '%05d' "$i"
  echo "${img_num}.png"

  python big_conv.py --gt_path "${gt_base_path}${img_num}.png" --gt_reso $gt_reso --sigma 0.005 \
  --psf_path $psf_path --psf_reso 820 --iters 10000 --mask_type "opt_radial"\
  --tv_alpha $tv_alpha_full  --save_path "results/big_conv/quantitative_analysis/opt_radial/${img_num}/"

  python big_conv.py --gt_path "${gt_base_path}${img_num}.png" --gt_reso $gt_reso --sigma 0.005 \
  --psf_path $psf_path --psf_reso 512 --psf_center_crop_size 128 --iters 10000 --mask_type "restricted" \
  --tv_alpha $tv_alpha_rest  --save_path "results/big_conv/quantitative_analysis/restricted6per/${img_num}/"
done