# noise_range=[]
# mse_range=[]

# for noise in noise_range
# for mse in mse_range
# done
# done
step=500
timestamp=$(date +%m-%d)

sampling_numbers='10'

for number in $sampling_numbers
do
echo $number

python projector.py \
 --f1_d 0 \
 --ckpt checkpoint/shapenet_val_size128_bs64_iter150000/142000.pt \
  --size 128 \
 --step $step \
 --id_aware \
 --inject_index 4 \
 --proj_latent inversion/Mar_06_20:55_1000_None_5/000003_4imgs.pt\
 --nerf_psnr_path=/mnt/lustre/yslan/Repo/NVS/Projects/nerf-pytorch/logs/PIXNERF/9views_viewdirs_raw__id0_instance_1_srn_car_traintest_resnet_6_0_viewasinput/test_psnr_epoch_010001.npy\
 --nerf_pred_path=/mnt/lustre/yslan/Repo/NVS/Projects/nerf-pytorch/logs/PIXNERF/9views_viewdirs_raw__id0_instance_1_srn_car_traintest_resnet_6_0_viewasinput/testset_010001\
 --w_l1 0\
 --sampling_range 50\
 --sampling_strategy uniform\
 --sampling_numbers $number\

done
echo all done

#  inversion/Mar_04_22:56/id_aware/000003_inject4_4imgs.pt \

# /mnt/lustre/yslan/Repo/NVS/Projects/Generative/Generation/deep-generative-prior/data/synthetic/5v/testset_199999/rgb/020.png \
# /mnt/lustre/yslan/Repo/NVS/Projects/Generative/Generation/deep-generative-prior/data/synthetic/5v/testset_199999/rgb/090.png \
# /mnt/lustre/yslan/Repo/NVS/Projects/Generative/Generation/deep-generative-prior/data/synthetic/5v/testset_199999/rgb/050.png \
# /mnt/lustre/yslan/Repo/NVS/Projects/Generative/Generation/deep-generative-prior/data/synthetic/5v/testset_199999/rgb/120.png \
# /mnt/lustre/yslan/Repo/NVS/Projects/Generative/Generation/deep-generative-prior/data/synthetic/5v/testset_199999/rgb/180.png \
#  inversion/id_aware/000006_inject4_7imgs.pt \
#  --no_noise_explore \
#  --ckpt checkpoint/carla_128/050000.pt \
#  ../graf/data/carla/512/009990.png  ../graf/data/carla/512/009991.png
#   --noise 0.1 \