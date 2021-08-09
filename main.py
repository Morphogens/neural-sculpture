from sdf_clip import SDFCLIP

sdf_clip = SDFCLIP(
    prompt="Mesh of a bunny rabbit rendered with zbrush maya",
    out_img_width=256,
    out_img_height=256,
    out_dir="./results",
)

sdf_clip.run(
    learning_rate=0.01,
    tolerance=8 / 10,
    num_iters_per_cam=1,
    num_iters_per_res=64,
    image_loss_weight=100,
    sdf_loss_weight=0.1,
    lp_loss_weight=0.1,
)

# sdf loss --> regulates that the shape is smooth
# lp loss --> regulates that the elements are altogether