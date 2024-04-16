# sd-webui-pag

simplified version of original repo to only contain the PAG algorithm

## COMPATIBILITY NOTICE

For [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) and [ComfyUI](github.com/comfyanonymous/ComfyUI), checkout [sd-perturbed-attention](https://github.com/pamparamm/sd-perturbed-attention)

---

## Perturbed Attention Guidance

[Arxiv page](https://arxiv.org/abs/2403.17377)

An alternative/complementary method to CFG (Classifier-Free Guidance) that increases sampling quality.

### Controls

* **PAG Scale**: Controls the intensity of effect of PAG on the generated image. It's recommended to set to around 3, but other values are also possible.

### Results

Prompt: "a puppy and a kitten on the moon"

* SD 1.5  
![image](./images/xyz_grid-3040-1-a%20puppy%20and%20a%20kitten%20on%20the%20moon.png)

* SD XL  
![image](./images/xyz_grid-3041-1-a%20puppy%20and%20a%20kitten%20on%20the%20moon.jpg)

### Also check out the paper authors' official project page

* [https://ku-cvlab.github.io/Perturbed-Attention-Guidance/](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/)
