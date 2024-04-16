import logging
from os import environ

import modules.scripts as scripts
from modules import script_callbacks

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))


"""

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def pag_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)

    return fun


def pag_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "pag_active"):
            setattr(p, "pag_active", True)
        setattr(p, field, x)

    return fun


def get_xyz_axis_options() -> set:
    xyz_grid = [
        x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"
    ][0].module
    extra_axis_options = {
        xyz_grid.AxisOption(
            "[PAG] Active",
            str,
            pag_apply_override("pag_active", boolean=True),
            choices=xyz_grid.boolean_choice(reverse=True),
        ),
        xyz_grid.AxisOption("[PAG] PAG Scale", float, pag_apply_field("pag_scale")),
        # xyz_grid.AxisOption("[PAG] ctnms_alpha", float, pag_apply_field("pag_ctnms_alpha")),
    }
    return extra_axis_options


def make_axis_options(extra_axis_options: set):
    xyz_grid = [
        x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"
    ][0].module
    current_opts = [x.label for x in xyz_grid.axis_options]
    # TODO:
    for opt in extra_axis_options:
        if opt.label in current_opts:
            return
    xyz_grid.axis_options.extend(extra_axis_options)


def callback_before_ui():
    try:
        extra_axis_options = get_xyz_axis_options()
        make_axis_options(extra_axis_options)
    except:
        logger.exception("Incantation: Error while making axis options")


script_callbacks.on_before_ui(callback_before_ui)
