#####################################################################
#####################################################################
####################### two pawns sanken toy ########################
#####################################################################
#####################################################################
meas_base_path="data/prototype/two_pawns_sanken_multiexposureV2/"
psf_base_path="data/prototype/psfV2/"
tv_alpha_full=0.0000001
lrate_full=1.0
tv_alpha_rest=0.0000005
lrate_rest=0.1

## TV2D parameter search
#for tv_alpha in 0.000005 0.00001 0.00005 0.0001 # 0.00000001 0.00000005 0.0000001 0.0000005 0.000001
#do
#    python big_conv.py --meas_path "${meas_base_path}/opt_radial_5000us.tiff" --meas_reso 512 \
#  --psf_path "${psf_base_path}/opt_radial/40cm.tiff" --psf_reso 820 --iters 10000 \
#  --tv_alpha $tv_alpha  --save_path "results/big_conv/two_pawns_sanken_multiexposureV2/opt_radial_tv_search/${tv_alpha}/"
#done

## PSF size search
#for psf_reso in 512 640 768 820 870 920 970 1024
#do
#python big_conv.py --meas_path "${meas_base_path}/opt_radial_5000us.tiff" --meas_reso 512 \
#--psf_path "${psf_base_path}/opt_radial/40cm.tiff" --psf_reso ${psf_reso} --iters 10000 \
#--tv_alpha $tv_alpha_full  --save_path "results/big_conv/two_pawns_sanken_multiexposureV2/opt_radial_psfreso_synthetic/reso_${psf_reso}/"
#done

# Synthetic PSF evaluation (check if synthetic PSF reconstructs prototype camera scenes)
#for psf_reso in 512 640 768 820 870 920 970 1024
#do
#python big_conv.py --meas_path "${meas_base_path}/opt_radial_5000us.tiff" --meas_reso 512 \
#--psf_path "data/simulation/opt_radial_mask.npy" --mask_type "opt_radial" --psf_reso ${psf_reso} --iters 10000 \
#--tv_alpha $tv_alpha_full  --save_path "results/big_conv/two_pawns_sanken_multiexposureV2/opt_radial_psfreso_synthetic/reso_${psf_reso}/"
#done

for exposure in 2000 5000 10000 15000
do
  python big_conv.py --meas_path "${meas_base_path}/opt_radial_${exposure}us.tiff" --meas_reso 512\
  --psf_path "${psf_base_path}/opt_radial/40cm.tiff" --psf_reso 820 --iters 10000 --lrate ${lrate_full} \
  --tv_alpha $tv_alpha_full  --save_path "results/big_conv/two_pawns_sanken_multiexposureV2/opt_radial/40cm/${exposure}us/" --redo 1

  python big_conv.py --meas_path "${meas_base_path}/restricted6per_${exposure}us.tiff" --meas_reso 512\
  --psf_path "${psf_base_path}/restricted6per/40cm.tiff" --psf_reso 512 --iters 10000 --lrate ${lrate_rest} \
  --tv_alpha $tv_alpha_rest  --save_path "results/big_conv/two_pawns_sanken_multiexposureV2/restricted6per/40cm/${exposure}us/" --redo 1
done

#####################################################################
#####################################################################
######################### FFHQ human faces ##########################
#####################################################################
#####################################################################
meas_base_path="data/prototype/monitor/FFHQ/"
psf_base_path="data/prototype/psfV2/"
tv_alpha_full=0.0000007
tv_alpha_rest=0.000003
lrate_full=1.0
lrate_rest=0.001

# PSF size search
img_num="00114"
for psf_reso in 512 640 768 820 870 920 970 1024
do
python big_conv.py --meas_path "${meas_base_path}/${img_num}/opt_radial_5000us.tiff" --meas_reso 512 \
--psf_path "${psf_base_path}/opt_radial/40cm.tiff" --psf_reso ${psf_reso} --iters 10000 \
--tv_alpha $tv_alpha_full  --save_path "results/big_conv/monitor/00114/opt_radial_psfreso/reso_${psf_reso}/"
done

# Reconstruction
for img_num in "00000" "00006" "00029" "00014" "00113" "00114"
do
  for exposure in 15000 10000 5000 2000
  do
    python big_conv.py --meas_path "${meas_base_path}/${img_num}/restricted6per_${exposure}us.tiff" --meas_reso 512\
    --psf_path "${psf_base_path}/restricted_6per/40cm.tiff" --psf_reso 512 --iters 10000 --lrate ${lrate_rest} \
    --tv_alpha $tv_alpha_rest  --save_path "results/big_conv/monitor/FFHQ/${img_num}/restricted6per/40cm/${exposure}us/" --redo 1

    python big_conv.py --meas_path "${meas_base_path}/${img_num}/opt_radial_${exposure}us.tiff" --meas_reso 512\
    --psf_path "${psf_base_path}/opt_radial/40cm.tiff" --psf_reso 820 --iters 10000  --lrate ${lrate_full} \
    --tv_alpha $tv_alpha_full  --save_path "results/big_conv/monitor/FFHQ/${img_num}/opt_radial/40cm/${exposure}us/" --redo 1

  done
done

