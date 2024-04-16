import logging
import typing as tg
from collections import OrderedDict
from os import environ
from warnings import warn

import gradio as gr
import torch

import modules.scripts as scripts
from modules import patches, script_callbacks, shared
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import (
    AfterCFGCallbackParams,
    CFGDenoisedParams,
    CFGDenoiserParams,
)
from modules.sd_samplers_cfg_denoiser import catenate_conds

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""
An unofficial implementation of "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance" for Automatic1111 WebUI.

@misc{ahn2024selfrectifying,
      title={Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance},
      author={Donghoon Ahn and Hyoungwon Cho and Jaewon Min and Wooseok Jang and Jungwoo Kim and SeonHwa Kim and Hyun Hee Park and Kyong Hwan Jin and Seungryong Kim},
      year={2024},
      eprint={2403.17377},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


class PAGStateParams:
    def __init__(self) -> None:
        self.pag_scale: float = -1  # PAG guidance scale
        self.guidance_scale: float = -1  # CFG
        self.x_in = None
        self.text_cond: dict | None = None
        self.image_cond: dict | None = None
        self.sigma = None
        self.text_uncond: dict | None = None
        self.make_condition_dict: tg.Callable | None = None  # callable lambda
        self.crossattn_modules: list = []  # callable lambda
        self.to_v_modules: list = []
        self.to_out_modules: list = []
        self.pag_x_out = None
        self.batch_size = -1  # Batch size
        self.denoiser = None  # CFGDenoiser
        self.patched_combine_denoised = None
        self.conds_list = None
        self.uncond_shape_0 = None


class PAGExtensionScript(scripts.Script):
    def __init__(self):
        self.cached_c = [None, None]
        self.handles = []

    # Extension title in menu UI
    def title(self) -> str:
        return "Perturbed Attention Guidance"

    # Decide to show menu in txt2img or img2img
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # Setup menu ui detail
    def ui(self, is_img2img) -> list:
        with gr.Accordion("Perturbed Attention Guidance", open=False):
            active = gr.Checkbox(
                value=False, default=False, label="Active", elem_id="pag_active"
            )
            with gr.Row():
                pag_scale = gr.Slider(
                    value=3.0,
                    minimum=0,
                    maximum=20.0,
                    step=0.5,
                    label="PAG Scale",
                    elem_id="pag_scale",
                    info="",
                )
        self.infotext_fields = [  # type: ignore
            (active, lambda d: gr.Checkbox.update(value="PAG Active" in d)),
            (pag_scale, "PAG Scale"),
        ]
        self.paste_field_names = [  # type: ignore
            "pag_active",
            "pag_scale",
        ]
        return [active, pag_scale]

    def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        self.pag_process_batch(p, *args, **kwargs)

    def pag_process_batch(
        self, p: StableDiffusionProcessing, active, pag_scale, *args, **kwargs
    ):
        # cleanup previous hooks always
        script_callbacks.remove_current_script_callbacks()
        self.remove_all_hooks()

        active = getattr(p, "pag_active", active)
        if active is False:
            return
        pag_scale = getattr(p, "pag_scale", pag_scale)

        p.extra_generation_params.update(
            {
                "PAG Active": active,
                "PAG Scale": pag_scale,
            }
        )
        self.create_hook(p, active, pag_scale)

    def create_hook(
        self, p: StableDiffusionProcessing, active, pag_scale, *args, **kwargs
    ):
        # Create a list of parameters for each concept
        pag_params = PAGStateParams()
        pag_params.pag_scale = pag_scale
        pag_params.guidance_scale = p.cfg_scale
        pag_params.batch_size = p.batch_size
        pag_params.denoiser = None

        # Get all the qv modules
        cross_attn_modules = self.get_cross_attn_modules()
        if len(cross_attn_modules) == 0:
            logger.error("No cross attention modules found, cannot proceed with PAG")
            return
        pag_params.crossattn_modules = [
            m for m in cross_attn_modules if "CrossAttention" in m.__class__.__name__
        ]

        # Use lambda to call the callback function with the parameters to avoid global variables
        cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(
            callback_params, pag_params
        )
        cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(
            callback_params, pag_params
        )
        # after_cfg_lambda = lambda x: self.cfg_after_cfg_callback(x, params)
        unhook_lambda = lambda _: self.unhook_callbacks(pag_params)

        self.ready_hijack_forward(pag_params.crossattn_modules, pag_scale)

        logger.debug("Hooked callbacks")
        script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
        script_callbacks.on_cfg_denoised(cfg_denoised_lambda)
        # script_callbacks.on_cfg_after_cfg(after_cfg_lambda)
        script_callbacks.on_script_unloaded(unhook_lambda)

    def postprocess_batch(self, p, *args, **kwargs):
        self.pag_postprocess_batch(p, *args, **kwargs)

    def pag_postprocess_batch(self, p, active, *args, **kwargs):
        script_callbacks.remove_current_script_callbacks()

        logger.debug("Removed script callbacks")
        active = getattr(p, "pag_active", active)
        if active is False:
            return

    def remove_all_hooks(self):
        cross_attn_modules = self.get_cross_attn_modules()
        for module in cross_attn_modules:
            to_v = getattr(module, "to_v", None)
            self.remove_field_cross_attn_modules(module, "pag_enable")
            self.remove_field_cross_attn_modules(module, "pag_last_to_v")
            _remove_all_forward_hooks(module, "pag_pre_hook")

            if to_v is not None:
                self.remove_field_cross_attn_modules(to_v, "pag_parent_module")
                _remove_all_forward_hooks(to_v, "to_v_pre_hook")

    def unhook_callbacks(self, pag_params: PAGStateParams):
        if pag_params is None:
            logger.error("PAG params is None")
            return

        if pag_params.denoiser is not None:
            denoiser = pag_params.denoiser
            setattr(denoiser, "combine_denoised_patched", False)
            try:
                patches.undo(__name__, denoiser, "combine_denoised")
            except KeyError:
                logger.exception("KeyError unhooking combine_denoised")
                pass
            except RuntimeError:
                logger.exception("RuntimeError unhooking combine_denoised")
                pass
            pag_params.denoiser = None

    def ready_hijack_forward(self, crossattn_modules, pag_scale):
        """Create hooks in the forward pass of the cross attention modules
        Copies the output of the to_v module to the parent module
        Then applies the PAG perturbation to the output of the cross attention module (multiplication by identity)
        """

        # add field for last_to_v
        for module in crossattn_modules:
            to_v = getattr(module, "to_v", None)
            self.add_field_cross_attn_modules(module, "pag_enable", False)
            self.add_field_cross_attn_modules(module, "pag_last_to_v", None)
            self.add_field_cross_attn_modules(to_v, "pag_parent_module", [module])
            # self.add_field_cross_attn_modules(to_out, 'pag_parent_module', [module])

        def to_v_pre_hook(module, input, kwargs, output):
            """Copy the output of the to_v module to the parent module"""
            parent_module = getattr(module, "pag_parent_module", None)
            # copy the output of the to_v module to the parent module

            if parent_module is not None:
                setattr(parent_module[0], "pag_last_to_v", output.detach().clone())

        def pag_pre_hook(module, input, kwargs, output):
            if (
                hasattr(module, "pag_enable")
                and getattr(module, "pag_enable", False) is False
            ):
                return
            if not hasattr(module, "pag_last_to_v"):
                # oops we forgot to unhook
                return

            batch_size, seq_len, inner_dim = output.shape
            identity = torch.eye(seq_len).expand(batch_size, -1, -1).to(shared.device)

            # get the last to_v output and save it
            last_to_v = getattr(module, "pag_last_to_v", None)
            if last_to_v is not None:
                new_output = torch.einsum("bij,bjk->bik", identity, last_to_v)
                return new_output
            else:
                # this is bad
                return output

        # Create hooks
        for module in crossattn_modules:
            module.register_forward_hook(pag_pre_hook, with_kwargs=True)

            to_v = getattr(module, "to_v", None)
            if to_v is not None:
                to_v.register_forward_hook(to_v_pre_hook, with_kwargs=True)

    def get_middle_block_modules(self):
        """Get all attention modules from the middle block
        Refere to page 22 of the PAG paper, Appendix A.2

        """
        try:
            m = shared.sd_model
            nlm = m.network_layer_mapping
            middle_block_modules = [
                m
                for m in nlm.values()
                if "middle_block_1_transformer_blocks_0_attn1" in m.network_layer_name
                and "CrossAttention" in m.__class__.__name__
            ]
            return middle_block_modules
        except AttributeError:
            logger.exception(
                "AttributeError in get_middle_block_modules", stack_info=True
            )
            return []
        except Exception:
            logger.exception("Exception in get_middle_block_modules", stack_info=True)
            return []

    def get_cross_attn_modules(self):
        """Get all cross attention modules"""
        return self.get_middle_block_modules()

    def add_field_cross_attn_modules(self, module, field, value):
        """Add a field to a module if it doesn't exist"""
        if not hasattr(module, field):
            setattr(module, field, value)

    def remove_field_cross_attn_modules(self, module, field):
        """Remove a field from a module if it exists"""
        if hasattr(module, field):
            delattr(module, field)

    def on_cfg_denoiser_callback(
        self, params: CFGDenoiserParams, pag_params: PAGStateParams
    ):
        # always unhook
        self.unhook_callbacks(pag_params)

        # patch combine_denoised
        if pag_params.denoiser is None:
            pag_params.denoiser = params.denoiser
        if getattr(params.denoiser, "combine_denoised_patched", False) is False:
            try:
                setattr(
                    params.denoiser,
                    "combine_denoised_original",
                    getattr(params.denoiser, "combine_denoised"),
                )
                # create patch that references the original function
                pass_conds_func = (
                    lambda *args, **kwargs: combine_denoised_pass_conds_list(
                        *args,
                        **kwargs,
                        original_func=getattr(
                            params.denoiser, "combine_denoised_original"
                        ),
                        pag_params=pag_params
                    )
                )
                pag_params.patched_combine_denoised = patches.patch(
                    __name__, params.denoiser, "combine_denoised", pass_conds_func
                )
                setattr(params.denoiser, "combine_denoised_patched", True)
                setattr(
                    params.denoiser,
                    "combine_denoised_original",
                    patches.original(__name__, params.denoiser, "combine_denoised"),
                )
            except KeyError:
                logger.exception("KeyError patching combine_denoised")
                pass
            except RuntimeError:
                logger.exception("RuntimeError patching combine_denoised")
                pass

        if isinstance(params.text_cond, dict):
            text_cond = params.text_cond["crossattn"]  # SD XL
            pag_params.text_cond = {}
            pag_params.text_uncond = {}
            for key, value in params.text_cond.items():
                pag_params.text_cond[key] = value.clone().detach()
                pag_params.text_uncond[key] = value.clone().detach()
        else:
            text_cond = params.text_cond  # SD 1.5
            pag_params.text_cond = text_cond.clone().detach()
            pag_params.text_uncond = text_cond.clone().detach()

        pag_params.x_in = params.x.clone().detach()
        pag_params.sigma = params.sigma.clone().detach()
        pag_params.image_cond = params.image_cond.clone().detach()
        pag_params.denoiser = params.denoiser
        pag_params.make_condition_dict = get_make_condition_dict_fn(params.text_uncond)

    def on_cfg_denoised_callback(
        self, params: CFGDenoisedParams, pag_params: PAGStateParams
    ):
        """Callback function for the CFGDenoisedParams
        Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine

        """
        # passed from on_cfg_denoiser_callback
        x_in = pag_params.x_in
        tensor = pag_params.text_cond
        uncond = pag_params.text_uncond
        image_cond_in = pag_params.image_cond
        sigma_in = pag_params.sigma

        # concatenate the conditions
        # "modules/sd_samplers_cfg_denoiser.py:237"
        cond_in = catenate_conds([tensor, uncond])
        make_condition_dict = get_make_condition_dict_fn(uncond)
        conds = make_condition_dict(cond_in, image_cond_in)

        # set pag_enable to True for the hooked cross attention modules
        for module in pag_params.crossattn_modules:
            setattr(module, "pag_enable", True)

        # get the PAG guidance (is there a way to optimize this so we don't have to calculate it twice?)
        pag_x_out = params.inner_model(x_in, sigma_in, cond=conds)

        # update pag_x_out
        pag_params.pag_x_out = pag_x_out

        # set pag_enable to False
        for module in pag_params.crossattn_modules:
            setattr(module, "pag_enable", False)

    def cfg_after_cfg_callback(
        self, params: AfterCFGCallbackParams, pag_params: PAGStateParams
    ):
        # self.unhook_callbacks(pag_params)
        pass


def combine_denoised_pass_conds_list(*args, **kwargs):
    """Hijacked function for combine_denoised in CFGDenoiser"""
    original_func = kwargs.get("original_func", None)
    new_params = kwargs.get("pag_params", None)

    if new_params is None:
        logger.error("new_params is None")
        return original_func(*args)

    def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0] :]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (
                    weight * cond_scale
                )
                try:
                    denoised[i] += (x_out[cond_index] - new_params.pag_x_out[i]) * (
                        weight * new_params.pag_scale
                    )
                except TypeError:
                    logger.exception("TypeError in combine_denoised_pass_conds_list")
                except IndexError:
                    logger.exception("IndexError in combine_denoised_pass_conds_list")
                # logger.debug(f"added PAG guidance to denoised - pag_scale:{global_scale}")
        return denoised

    return new_combine_denoised(*args)


# from modules/sd_samplers_cfg_denoiser.py:187-195
def get_make_condition_dict_fn(text_uncond):
    if shared.sd_model.model.conditioning_key == "crossattn-adm":
        make_condition_dict = lambda c_crossattn, c_adm: {
            "c_crossattn": [c_crossattn],
            "c_adm": c_adm,
        }
    else:
        if isinstance(text_uncond, dict):
            make_condition_dict = lambda c_crossattn, c_concat: {
                **c_crossattn,
                "c_concat": [c_concat],
            }
        else:
            make_condition_dict = lambda c_crossattn, c_concat: {
                "c_crossattn": [c_crossattn],
                "c_concat": [c_concat],
            }
    return make_condition_dict


# thanks torch; removing hooks DOESN'T WORK
# thank you to @ProGamerGov for this https://github.com/pytorch/pytorch/issues/70455
def _remove_all_forward_hooks(
    module: torch.nn.Module, hook_fn_name: str | None = None
) -> None:
    """
    This function removes all forward hooks in the specified module, without requiring
    any hook handles. This lets us clean up & remove any hooks that weren't property
    deleted.

    Warning: Various PyTorch modules and systems make use of hooks, and thus extreme
    caution should be exercised when removing all hooks. Users are recommended to give
    their hook function a unique name that can be used to safely identify and remove
    the target forward hooks.

    Args:

        module (nn.Module): The module instance to remove forward hooks from.
        hook_fn_name (str, optional): Optionally only remove specific forward hooks
            based on their function's __name__ attribute.
            Default: None
    """

    if hook_fn_name is None:
        warn("Removing all active hooks can break some PyTorch modules & systems.")

    def _remove_hooks(m: torch.nn.Module, name: str | None = None) -> None:
        if hasattr(module, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                if name is not None:
                    dict_items = list(m._forward_hooks.items())
                    m._forward_hooks = OrderedDict(
                        [(i, fn) for i, fn in dict_items if fn.__name__ != name]
                    )
                else:
                    m._forward_hooks = OrderedDict()

    def _remove_child_hooks(
        target_module: torch.nn.Module, hook_name: str | None = None
    ) -> None:
        for name, child in target_module._modules.items():
            if child is not None:
                _remove_hooks(child, hook_name)
                _remove_child_hooks(child, hook_name)

    # Remove hooks from target submodules
    _remove_child_hooks(module, hook_fn_name)

    # Remove hooks from the target module
    _remove_hooks(module, hook_fn_name)
