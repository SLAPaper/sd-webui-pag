import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
from PIL import Image
import numpy as np

from modules import script_callbacks, prompt_parser
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
from modules.prompt_parser import reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing, decode_latent_batch
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import pad_cond
from modules import shared, devices

import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""

An unofficial implementation of Seek for Incantations: Towards Accurate Text-to-Image Diffusion Synthesis
through Prompt Engineering for Automatic1111 WebUI

@misc{yu2024seek,
      title={Seek for Incantations: Towards Accurate Text-to-Image Diffusion Synthesis through Prompt Engineering}, 
      author={Chang Yu and Junran Peng and Xiangyu Zhu and Zhaoxiang Zhang and Qi Tian and Zhen Lei},
      year={2024},
      eprint={2401.06345},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""

class IncantStateParams:
        def __init__(self):
                self.coarse = 10
                self.fine = 30

class IncantExtensionScript(scripts.Script):
        def __init__(self):
                self.cached_c = [None, None]

        # Extension title in menu UI
        def title(self):
                return "Incantations"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('Incantations', open=False):
                        active = gr.Checkbox(value=True, default=True, label="Active", elem_id='incant_active')
                        quality = gr.Checkbox(value=True, default=True, label="Quality Guidance", elem_id='incant_quality')
                        with gr.Row():
                                coarse_step = gr.Slider(value = 10, minimum = 0, maximum = 100, step = 1, label="Coarse Step", elem_id = 'incant_coarse')
                                fine_step = gr.Slider(value = 30, minimum = 0, maximum = 100, step = 1, label="Fine Step", elem_id = 'incant_fine')
                        # with gr.Row():
                        #         warmup = gr.Slider(value = 10, minimum = 0, maximum = 30, step = 1, label="Warmup Period", elem_id = 'incant_warmup', info="How many steps to wait before applying semantic guidance, default 10")
                        #         edit_guidance_scale = gr.Slider(value = 1.0, minimum = 0.0, maximum = 20.0, step = 0.01, label="Edit Guidance Scale", elem_id = 'incant_edit_guidance_scale', info="Scale of edit guidance, default 1.0")
                        #         tail_percentage_threshold = gr.Slider(value = 0.05, minimum = 0.0, maximum = 1.0, step = 0.01, label="Tail Percentage Threshold", elem_id = 'incant_tail_percentage_threshold', info="The percentage of latents to modify, default 0.05")
                        #         momentum_scale = gr.Slider(value = 0.3, minimum = 0.0, maximum = 1.0, step = 0.01, label="Momentum Scale", elem_id = 'incant_momentum_scale', info="Scale of momentum, default 0.3")
                        #         momentum_beta = gr.Slider(value = 0.6, minimum = 0.0, maximum = 0.999, step = 0.01, label="Momentum Beta", elem_id = 'incant_momentum_beta', info="Beta for momentum, default 0.6")
                active.do_not_save_to_config = True
                quality.do_not_save_to_config = True
                coarse_step.do_not_save_to_config = True
                fine_step.do_not_save_to_config = True
                # self.infotext_fields = [
                #         (active, lambda d: gr.Checkbox.update(value='INCANT Active' in d)),
                #         (coarse_step, 'INCANT Prompt'),
                #         (fine_step, 'INCANT Negative Prompt'),
                # ]
                # self.paste_field_names = [
                #         'incant_active',
                #         'incant_quality',
                #         'incant_coarse',
                #         'incant_fine',
                # ]
                return [active, quality, coarse_step, fine_step]

        def process_batch(self, p: StableDiffusionProcessing, active, quality, coarse_step, fine_step, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                quality = getattr(p, "incant_quality", quality)
                coarse_step = getattr(p, "incant_coarse", coarse_step)
                fine_step = getattr(p, "incant_fine", fine_step)

                sd_model = shared.sd_model

                p.extra_generation_params = {
                        "INCANT Active": active,
                        "INCANT Quality": quality,
                        "INCANT Coarse": coarse_step,
                        "INCANT Fine": fine_step,
                }

                self.create_hook(p, active, quality, coarse_step, fine_step, *args, **kwargs)

        def parse_concept_prompt(self, prompt:str) -> list[str]:
                """
                Separate prompt by comma into a list of concepts
                TODO: parse prompt into a list of concepts using A1111 functions
                >>> g = lambda prompt: self.parse_concept_prompt(prompt)
                >>> g("")
                []
                >>> g("apples")
                ['apples']
                >>> g("apple, banana, carrot")
                ['apple', 'banana', 'carrot']
                """
                if len(prompt) == 0:
                        return []
                return [x.strip() for x in prompt.split(",")]

        def create_hook(self, p, active, quality, coarse, fine, *args, **kwargs):
                # Create a list of parameters for each concept
                incant_params = IncantStateParams()
                incant_params.coarse = coarse
                incant_params.fine = fine 

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, incant_params)
                y2 = lambda params: self.on_cfg_denoised_callback(params, incant_params)
                y3 = lambda params: self.cfg_after_cfg_callback(params, incant_params)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_cfg_denoised(y2)
                script_callbacks.cfg_after_cfg_callback(y3)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)
        
        def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams, incant_params):
                active = getattr(p, "incant_active", active)
                pass

        def postprocess_batch(self, p, active, quality, coarse, fine, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                script_callbacks.remove_current_script_callbacks()

        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, incant_params: IncantStateParams):
                coarse = incant_params.coarse
                fine= incant_params.coarse
                x = params.x
                step = params.sampling_step
                max_step = params.total_sampling_steps
                batch_images = []
                if step == coarse:
                        # decode the latent
                        x_samples_ddim = decode_latent_batch(shared.sd_model, x, target_device=devices.cpu, check_for_nans=True)
                        x_samples_ddim = torch.stack(x_samples_ddim).float()
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        for i, x_sample in enumerate(x_samples_ddim):
                                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                                x_sample = x_sample.astype(np.uint8)
                                image = Image.fromarray(x_sample)
                                batch_images.append(image)
                                #image.show()

                #decoded_samples = torch.from_numpy(np.array(batch_images))
                #decoded_samples = decoded_samples.to(shared.device, dtype=devices.dtype_vae)

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, incant_params: IncantStateParams):
                # TODO: add option to opt out of batching for performance
                sampling_step = params.sampling_step
                text_cond = params.text_cond
                text_uncond = params.text_uncond
        

# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def incant_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def incant_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "incant_active"):
                setattr(p, "incant_active", True)
        setattr(p, field, x)

    return fun

def make_axis_options():
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
        extra_axis_options = {
                xyz_grid.AxisOption("[Incant] Active", str, incant_apply_override('incant_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[Incant] Quality", str, incant_apply_override('incant_quality', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[Incant] Coarse Step", int, incant_apply_field("incant_coarse")),
                xyz_grid.AxisOption("[Incant] Fine Step", int, incant_apply_field("incant_fine")),
                #xyz_grid.AxisOption("[Incant] Warmup Steps", int, incant_apply_field("incant_warmup")),
                #xyz_grid.AxisOption("[Incant] Guidance Scale", float, incant_apply_field("incant_edit_guidance_scale")),
                #xyz_grid.AxisOption("[Incant] Tail Percentage Threshold", float, incant_apply_field("incant_tail_percentage_threshold")),
                #xyz_grid.AxisOption("[Incant] Momentum Scale", float, incant_apply_field("incant_momentum_scale")),
                #xyz_grid.AxisOption("[Incant] Momentum Beta", float, incant_apply_field("incant_momentum_beta")),
        }
        if not any("[Incant]" in x.label for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
        try:
                make_axis_options()
        except:
                logger.exception("Incantation: Error while making axis options")

script_callbacks.on_before_ui(callback_before_ui)
