python projector.py \
 --ckpt checkpoint/shapenet_val_size128_bs64_iter150000/142000.pt \
  --size 128 \
 --step 1000 \
 --id_aware \
 --inject_index 4 \
 --noise 0.03 \
 --w_l1 1 \
 inversion/GT/shapenet/100715345ee54d7ae38b52b4ee9d36a3/rgb/000000.png \
 inversion/GT/shapenet/100715345ee54d7ae38b52b4ee9d36a3/rgb/000001.png \
 inversion/GT/shapenet/100715345ee54d7ae38b52b4ee9d36a3/rgb/000002.png \
 inversion/GT/shapenet/100715345ee54d7ae38b52b4ee9d36a3/rgb/000003.png \
#  inversion/GT/shapenet/100715345ee54d7ae38b52b4ee9d36a3/rgb/000004.png \
#  inversion/GT/shapenet/100715345ee54d7ae38b52b4ee9d36a3/rgb/000006.png \
#  inversion/GT/shapenet/100715345ee54d7ae38b52b4ee9d36a3/rgb/000007.png \

  # inversion/GT/shapenet/000000.png \
  # inversion/GT/shapenet/000002.png \
  # inversion/GT/shapenet/000004.png \
  # inversion/GT/shapenet/000006.png \
  # inversion/GT/shapenet/000012.png \
  # inversion/GT/shapenet/000016.png \
# /mnt/lustre/yslan/Repo/NVS/Projects/Generative/Generation/deep-generative-prior/data/synthetic/5v/train/rgb/000100.png \
# /mnt/lustre/yslan/Repo/NVS/Projects/Generative/Generation/deep-generative-prior/data/synthetic/5v/train/rgb/000000.png \
#  --no_noise_explore \

#  --ckpt checkpoint/carla_128/050000.pt \
#  ../graf/data/carla/512/009990.png  ../graf/data/carla/512/009991.png
#   --noise 0.1 \