import uuid
import os
import PIL.Image as Image
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision


def make_unique_experiment_path(base_dir: str) -> str:
    """
    Create a unique directory in the base directory, named as the least unused number.
    return: path to the unique directory
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # List all existing directories
    existing_dirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Convert directory names to integers, filter out non-numeric names
    existing_numbers = sorted([int(d) for d in existing_dirs if d.isdigit()])

    # Find the least unused number
    experiment_id = 1
    for number in existing_numbers:
        if number != experiment_id:
            break
        experiment_id += 1

    # Create the new directory
    experiment_output_path = os.path.join(base_dir, str(experiment_id))
    os.makedirs(experiment_output_path)

    return experiment_output_path


def get_processed_image(image_dir: str, device, resolution) -> torch.Tensor:
    src_img = Image.open(image_dir)
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)

    h, w = src_img.shape[-2:]
    src_img_512 = torchvision.transforms.functional.pad(
        src_img, ((resolution - w) // 2,), fill=0, padding_mode="constant"
    )
    input_image = F.interpolate(
        src_img, (resolution, resolution), mode="bilinear", align_corners=False
    )
    # drop alpha channel if it exists
    if input_image.shape[1] == 4:
        input_image = input_image[:, :3]

    return input_image


def process_image(image, device, resolution) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    src_img = image
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)

    h, w = src_img.shape[-2:]
    src_img_512 = torchvision.transforms.functional.pad(
        src_img, ((resolution - w) // 2,), fill=0, padding_mode="constant"
    )
    input_image = F.interpolate(
        src_img, (resolution, resolution), mode="bilinear", align_corners=False
    )
    # drop alpha channel if it exists
    if input_image.shape[1] == 4:
        input_image = input_image[:, :3]

    return input_image


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(42)


def dump_tensor(tensor, filename):
    with open(filename) as f:
        torch.save(tensor, f)
