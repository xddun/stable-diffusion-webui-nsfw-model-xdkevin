import os.path

import numpy as np
import torch
from PIL import Image
from diffusers.utils import logging
from modules import scripts
from transformers import AutoFeatureExtractor

from scripts.safety_checker import StableDiffusionSafetyChecker

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
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image,
        clip_input=safety_checker_input.pixel_values,
        safety_checker_adj=0,  # TODO: customize adjustment
    )

    return x_checked_image, has_nsfw_concept


def censor_batch(x):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy)
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
                assert y.shape == x.shape
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
        images = kwargs['images']
        images[:] = censor_batch(images)[:]
