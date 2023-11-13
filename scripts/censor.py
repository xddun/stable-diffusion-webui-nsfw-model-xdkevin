import copy
import os

import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image
from modules import scripts, shared

import logging
import sys


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("nsfw_xiedong")
logger.propagate = False
# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# Configure logger
loglevel_string = getattr(shared.cmd_opts, "controlnet_loglevel", "INFO")
loglevel = getattr(logging, loglevel_string.upper(), "info")
logger.setLevel(loglevel)

# Load ONNX model
onnx_model_path = os.path.join("extensions", "stable-diffusion-webui-nsfw-model-xdkevin", "scripts", "model_fp16.onnx")
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def safety_checker_xd(images):
    global ort_session, input_name, output_name
    pil_images = numpy_to_pil(images)
    has_nsfw_concepts = []
    for idx, image in enumerate(pil_images):
        cv2_img = np.array(image)
        # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2_img = cv2.resize(cv2_img, (299, 299))
        cv2_img = cv2_img.astype(np.float16) / 255.0
        cv2_img = np.expand_dims(cv2_img, axis=0)
        # Run the ONNX model
        res_list = ort_session.run([output_name], {input_name: cv2_img})[0].tolist()[0]
        logger.warning(f"Results for images {idx}: {res_list}")
        # print(res_list)
        # 打印最大值的下标和名称
        index = np.argmax(res_list)
        # cls_name = ["drawings", "hentai", "neutral", "porn", "sexy"]

        if index == 1 and res_list[index] > 0.5:
            has_nsfw_concepts.append(True)
        elif index == 3 and res_list[index] > 0.5:
            has_nsfw_concepts.append(True)
        elif index == 4 and res_list[index] > 0.6:
            has_nsfw_concepts.append(True)
        else:
            has_nsfw_concepts.append(False)

    if any(has_nsfw_concepts):
        logger.warning(
            "Potential NSFW content was detected in one or more images. A black image will be returned instead."
            " Try again with a different prompt and/or seed."
        )
    for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
        if has_nsfw_concept:
            images[idx] = np.zeros(images[idx].shape)
    return images, has_nsfw_concepts


# check and replace nsfw content
def check_safety(x_image):
    x_checked_image, has_nsfw_concept = safety_checker_xd(images=x_image)
    return x_checked_image, has_nsfw_concept


def censor_batch(x):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
    return x


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "xd NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']
        images[:] = censor_batch(images)[:]

