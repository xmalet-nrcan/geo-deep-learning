"""Models for deep learning."""

# install instructions for MultiScaleDeformableAttention:
# git clone https://github.com/facebookresearch/dinov3.git
# cd dinov3/dinov3/eval/segmentation/models/utils/ops
# pip install .

# install cuda toolkit if not already installed in env:
# for example, if using conda:
# conda install cuda-toolkit=12.1 -c nvidia


# new workflow to install MultiScaleDeformableAttention (uv + conda)

# source /opt/anaconda3/etc/profile.d/conda.sh

# conda create -n cuda-toolkit python=3.12
# conda activate cuda-toolkit
# conda install cuda-toolkit=12.8 -c nvidia

# conda activate cuda-toolkit
# uv pip install --no-build-isolation \
#   --python /home/valhassa/Projects/geo-deep-learning/.venv/bin/python \
#   ~/Projects/dinov3/dinov3/eval/segmentation/models/utils/ops
