import os.path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers.utils import logging
from scripts.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from modules import scripts

logger = logging.get_logger(__name__)

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None

warning_image = os.path.join("extensions", "stable-diffusion-webui-nsfw-filter", "warning", "warning.png")


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# check and replace nsfw content
def check_safety(x_image, safety_checker_adj: float):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image,
        clip_input=safety_checker_input.pixel_values,
        safety_checker_adj=safety_checker_adj,  # customize adjustment
    )

    return x_checked_image, has_nsfw_concept


def censor_batch(x, safety_checker_adj: float):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy, safety_checker_adj)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

    index = 0
    for unsafe_value in has_nsfw_concept:
        try:
            if unsafe_value is True:
                hwc = x.shape
                y = Image.open(warning_image).convert("RGB").resize((hwc[3], hwc[2]))
                y = (np.array(y) / 255.0).astype("float32")
                y = torch.from_numpy(y)
                y = torch.unsqueeze(y, 0).permute(0, 3, 1, 2)
                x[index] = y
            index += 1
        except Exception as e:
            logger.warning(e)
            index += 1

    return x


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Args:
            p:
            *args:
                args[0]: enable_nsfw_filer. True: NSFW filter enabled; False: NSFW filter disabled
                args[1]: safety_checker_adj
            **kwargs:
        Returns:
            images
        """

        images = kwargs['images']
        if args[0] is True:
            images[:] = censor_batch(images, args[1])[:]

    def ui(self, is_img2img):
        enable_nsfw_filer = gr.Checkbox(label='Enable NSFW filter',
                                        value=False,
                                        elem_id=self.elem_id("enable_nsfw_filer"))
        safety_checker_adj = gr.Slider(label="Safety checker adjustment",
                                       minimum=-0.5, maximum=0.5, value=0.0, step=0.001,
                                       elem_id=self.elem_id("safety_checker_adj"))
        return [enable_nsfw_filer, safety_checker_adj]
