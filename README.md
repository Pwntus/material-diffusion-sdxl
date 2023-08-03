# A Stable Diffusion XL Cog model that outputs tileable images for use in 3D applications such as [Monaverse](https://monaverse.com)

[![Replicate](https://replicate.com/pwntus/material-diffusion-sdxl/badge)](https://replicate.com/pwntus/material-diffusion-sdxl)

This is an implementation of the [Diffusers Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights.py

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"
