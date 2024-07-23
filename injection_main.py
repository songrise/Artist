# %%
import argparse, os


import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import argparse
import PIL.Image as Image
from torchvision.utils import make_grid
import numpy
from diffusers.schedulers import DDIMScheduler
import torch.nn.functional as F
from models import attn_injection
from omegaconf import OmegaConf
from typing import List, Tuple

import omegaconf
import utils.exp_utils
import json

device = torch.device("cuda")


def _get_text_embeddings(prompt: str, tokenizer, text_encoder, device):
    # Tokenize text and get embeddings
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=True,
        )

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    if prompt == "":
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        return negative_prompt_embeds, negative_pooled_prompt_embeds
    return prompt_embeds, pooled_prompt_embeds


def _encode_text_sdxl(model: StableDiffusionXLPipeline, prompt: str):
    device = model._execution_device
    (
        prompt_embeds,
        pooled_prompt_embeds,
    ) = _get_text_embeddings(prompt, model.tokenizer, model.text_encoder, device)
    (
        prompt_embeds_2,
        pooled_prompt_embeds_2,
    ) = _get_text_embeddings(prompt, model.tokenizer_2, model.text_encoder_2, device)
    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
    text_encoder_projection_dim = model.text_encoder_2.config.projection_dim
    add_time_ids = model._get_add_time_ids(
        (1024, 1024), (0, 0), (1024, 1024), torch.float16, text_encoder_projection_dim
    ).to(device)
    # repeat the time ids for each prompt
    add_time_ids = add_time_ids.repeat(len(prompt), 1)
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds_2,
        "time_ids": add_time_ids,
    }
    return added_cond_kwargs, prompt_embeds


def _encode_text_sdxl_with_negative(
    model: StableDiffusionXLPipeline, prompt: List[str]
):

    B = len(prompt)
    added_cond_kwargs, prompt_embeds = _encode_text_sdxl(model, prompt)
    added_cond_kwargs_uncond, prompt_embeds_uncond = _encode_text_sdxl(
        model, ["" for _ in range(B)]
    )
    prompt_embeds = torch.cat(
        (
            prompt_embeds_uncond,
            prompt_embeds,
        )
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat(
            (added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])
        ),
        "time_ids": torch.cat(
            (added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])
        ),
    }
    return added_cond_kwargs, prompt_embeds


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    intermediate_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):
    negative_prompt = [""] * len(prompt)
    # Encode prompt
    if isinstance(pipe, StableDiffusionPipeline):
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        added_cond_kwargs = None
    elif isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(
            pipe, prompt
        )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    latents = latents.repeat(len(prompt), 1, 1, 1)
    # assume that the first latent is used for reconstruction
    for i in tqdm(range(start_step, num_inference_steps)):
        latents[0] = intermediate_latents[(-i + 1)]
        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


# Sample function (regular DDIM), but disentangle the content and style
@torch.no_grad()
def sample_disentangled(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    intermediate_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    use_content_anchor=True,
    negative_prompt="",
    device=device,
):
    negative_prompt = [""] * len(prompt)
    vae_decoder = VaeImageProcessor(vae_scale_factor=pipe.vae.config.scaling_factor)
    # Encode prompt
    if isinstance(pipe, StableDiffusionPipeline):
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        added_cond_kwargs = None
    elif isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(
            pipe, prompt
        )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # save

    latent_shape = (
        (1, 4, 64, 64) if isinstance(pipe, StableDiffusionPipeline) else (1, 4, 64, 64)
    )
    generative_latent = torch.randn(latent_shape, device=device)
    generative_latent *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    latents = latents.repeat(len(prompt), 1, 1, 1)
    # randomly initalize the 1st lantent for generation

    latents[1] = generative_latent
    # assume that the first latent is used for reconstruction
    for i in tqdm(range(start_step, num_inference_steps), desc="Stylizing"):

        if use_content_anchor:
            latents[0] = intermediate_latents[(-i + 1)]
        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Post-processing
        # images = vae_decoder.postprocess(latents)
    pipe.vae.to(dtype=torch.float32)
    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    latents = 1 / pipe.vae.config.scaling_factor * latents
    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = pipe.numpy_to_pil(images)
    if isinstance(pipe, StableDiffusionXLPipeline):
        pipe.vae.to(dtype=torch.float16)

    return images


## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    if isinstance(pipe, StableDiffusionPipeline):
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        added_cond_kwargs = None
        latents = start_latents.clone().detach()
    elif isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(
            pipe, [prompt]
        )  # Latents are now the specified start latents
        latents = start_latents.clone().detach().half()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(
        range(1, num_inference_steps),
        total=num_inference_steps - 1,
        desc="DDIM Inversion",
    ):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
            alpha_t_next.sqrt() / alpha_t.sqrt()
        ) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def style_image_with_inversion(
    pipe,
    input_image,
    input_image_prompt,
    style_prompt,
    num_steps=100,
    start_step=30,
    guidance_scale=3.5,
    disentangle=False,
    share_attn=False,
    share_cross_attn=False,
    share_resnet_layers=[0, 1],
    share_attn_layers=[],
    c2s_layers=[0, 1],
    share_key=True,
    share_query=True,
    share_value=False,
    use_adain=True,
    use_content_anchor=True,
    output_dir: str = None,
    resnet_mode: str = None,
    return_intermediate=False,
    intermediate_latents=None,
):
    with torch.no_grad():
        pipe.vae.to(dtype=torch.float32)
        latent = pipe.vae.encode(input_image.to(device) * 2 - 1)
        # latent = pipe.vae.encode(input_image.to(device))
        l = pipe.vae.config.scaling_factor * latent.latent_dist.sample()
        if isinstance(pipe, StableDiffusionXLPipeline):
            pipe.vae.to(dtype=torch.float16)
    if intermediate_latents is None:
        inverted_latents = invert(
            pipe, l, input_image_prompt, num_inference_steps=num_steps
        )
    else:
        inverted_latents = intermediate_latents

    attn_injection.register_attention_processors(
        pipe,
        base_dir=output_dir,
        resnet_mode=resnet_mode,
        attn_mode="artist" if disentangle else "pnp",
        disentangle=disentangle,
        share_resblock=True,
        share_attn=share_attn,
        share_cross_attn=share_cross_attn,
        share_resnet_layers=share_resnet_layers,
        share_attn_layers=share_attn_layers,
        share_key=share_key,
        share_query=share_query,
        share_value=share_value,
        use_adain=use_adain,
        c2s_layers=c2s_layers,
    )

    if disentangle:
        final_im = sample_disentangled(
            pipe,
            style_prompt,
            start_latents=inverted_latents[-(start_step + 1)][None],
            intermediate_latents=inverted_latents,
            start_step=start_step,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            use_content_anchor=use_content_anchor,
        )
    else:
        final_im = sample(
            pipe,
            style_prompt,
            start_latents=inverted_latents[-(start_step + 1)][None],
            intermediate_latents=inverted_latents,
            start_step=start_step,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    # unset the attention processors
    attn_injection.unset_attention_processors(
        pipe,
        unset_share_attn=True,
        unset_share_resblock=True,
    )
    if return_intermediate:
        return final_im, inverted_latents
    return final_im


if __name__ == "__main__":

    # Load a pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base"
    ).to(device)

    # pipe = DiffusionPipeline.from_pretrained(
    #     # "playgroundai/playground-v2-1024px-aesthetic",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     add_watermarker=False,
    #     variant="fp16",
    # )
    # pipe.to("cuda")

    # Set up a DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    parser = argparse.ArgumentParser(description="Stable Diffusion with OmegaConf")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=["dataset", "cli", "app"],
        help="Path to the config file",
    )
    parser.add_argument(
        "--image_dir", type=str, default="test.png", help="Path to the image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="an impressionist painting",
        help="Stylization prompt",
    )
    # mode = "single_control_content"
    args = parser.parse_args()
    config_dir = args.config
    mode = args.mode
    # mode = "dataset"
    out_name = ["content_delegation", "style_delegation", "style_out"]

    if mode == "dataset":
        cfg = OmegaConf.load(config_dir)

        base_output_path = cfg.out_path
        if not os.path.exists(cfg.out_path):
            os.makedirs(cfg.out_path)
        base_output_path = os.path.join(base_output_path, cfg.exp_name)

        experiment_output_path = utils.exp_utils.make_unique_experiment_path(
            base_output_path
        )

        # Save the experiment configuration
        config_file_path = os.path.join(experiment_output_path, "config.yaml")
        omegaconf.OmegaConf.save(cfg, config_file_path)

        # Seed all

        annotation = json.load(open(cfg.annotation))
        with open(os.path.join(experiment_output_path, "annotation.json"), "w") as f:
            json.dump(annotation, f)
        for i, entry in enumerate(annotation):
            utils.exp_utils.seed_all(cfg.seed)
            image_path = entry["image_path"]
            src_prompt = entry["source_prompt"]
            tgt_prompt = entry["target_prompt"]
            resolution = 512 if isinstance(pipe, StableDiffusionXLPipeline) else 512
            input_image = utils.exp_utils.get_processed_image(
                image_path, device, resolution
            )

            prompt_in = [
                src_prompt,  # reconstruction
                tgt_prompt,  # uncontrolled style
                "",  # controlled style
            ]

            imgs = style_image_with_inversion(
                pipe,
                input_image,
                src_prompt,
                style_prompt=prompt_in,
                num_steps=cfg.num_steps,
                start_step=cfg.start_step,
                guidance_scale=cfg.style_cfg_scale,
                disentangle=cfg.disentangle,
                resnet_mode=cfg.resnet_mode,
                share_attn=cfg.share_attn,
                share_cross_attn=cfg.share_cross_attn,
                share_resnet_layers=cfg.share_resnet_layers,
                share_attn_layers=cfg.share_attn_layers,
                share_key=cfg.share_key,
                share_query=cfg.share_query,
                share_value=cfg.share_value,
                use_content_anchor=cfg.use_content_anchor,
                use_adain=cfg.use_adain,
                output_dir=experiment_output_path,
            )

            for j, img in enumerate(imgs):
                img.save(f"{experiment_output_path}/out_{i}_{out_name[j]}.png")
                print(
                    f"Image saved as {experiment_output_path}/out_{i}_{out_name[j]}.png"
                )
    elif mode == "cli":
        cfg = OmegaConf.load(config_dir)
        utils.exp_utils.seed_all(cfg.seed)
        image = utils.exp_utils.get_processed_image(args.image_dir, device, 512)
        tgt_prompt = args.prompt
        src_prompt = ""
        prompt_in = [
            "",  # reconstruction
            tgt_prompt,  # uncontrolled style
            "",  # controlled style
        ]
        out_dir = "./out"
        os.makedirs(out_dir, exist_ok=True)
        imgs = style_image_with_inversion(
            pipe,
            image,
            src_prompt,
            style_prompt=prompt_in,
            num_steps=cfg.num_steps,
            start_step=cfg.start_step,
            guidance_scale=cfg.style_cfg_scale,
            disentangle=cfg.disentangle,
            resnet_mode=cfg.resnet_mode,
            share_attn=cfg.share_attn,
            share_cross_attn=cfg.share_cross_attn,
            share_resnet_layers=cfg.share_resnet_layers,
            share_attn_layers=cfg.share_attn_layers,
            share_key=cfg.share_key,
            share_query=cfg.share_query,
            share_value=cfg.share_value,
            use_content_anchor=cfg.use_content_anchor,
            use_adain=cfg.use_adain,
            output_dir=out_dir,
        )
        image_base_name = os.path.basename(args.image_dir).split(".")[0]
        for j, img in enumerate(imgs):
            img.save(f"{out_dir}/{image_base_name}_out_{out_name[j]}.png")
            print(f"Image saved as {out_dir}/{image_base_name}_out_{out_name[j]}.png")
    elif mode == "app":
        # gradio
        import gradio as gr

        def style_transfer_app(
            prompt,
            image,
            cfg_scale=7.5,
            num_content_layers=4,
            num_style_layers=9,
            seed=0,
            progress=gr.Progress(track_tqdm=True),
        ):
            utils.exp_utils.seed_all(seed)
            image = utils.exp_utils.process_image(image, device, 512)

            tgt_prompt = prompt
            src_prompt = ""
            prompt_in = [
                "",  # reconstruction
                tgt_prompt,  # uncontrolled style
                "",  # controlled style
            ]

            share_resnet_layers = (
                list(range(num_content_layers)) if num_content_layers != 0 else None
            )
            share_attn_layers = (
                list(range(num_style_layers)) if num_style_layers != 0 else None
            )
            imgs = style_image_with_inversion(
                pipe,
                image,
                src_prompt,
                style_prompt=prompt_in,
                num_steps=50,
                start_step=0,
                guidance_scale=cfg_scale,
                disentangle=True,
                resnet_mode="hidden",
                share_attn=True,
                share_cross_attn=True,
                share_resnet_layers=share_resnet_layers,
                share_attn_layers=share_attn_layers,
                share_key=True,
                share_query=True,
                share_value=False,
                use_content_anchor=True,
                use_adain=True,
                output_dir="./",
            )

            return imgs[2]

        # load examples
        examples = []
        annotation = json.load(open("data/example/annotation.json"))
        for entry in annotation:
            image = utils.exp_utils.get_processed_image(
                entry["image_path"], device, 512
            )
            image = transforms.ToPILImage()(image[0])

            examples.append([entry["target_prompt"], image, None, None, None])

        text_input = gr.Textbox(
            value="An impressionist painting",
            label="Text Prompt",
            info="Describe the style you want to apply to the image, do not include the description of the image content itself",
            lines=2,
            placeholder="Enter a text prompt",
        )
        image_input = gr.Image(
            height="80%",
            width="80%",
            label="Content image (will be resized to 512x512)",
            interactive=True,
        )
        cfg_slider = gr.Slider(
            0,
            15,
            value=7.5,
            label="Classifier Free Guidance (CFG) Scale",
            info="higher values give more style, 7.5 should be good for most cases",
        )
        content_slider = gr.Slider(
            0,
            9,
            value=4,
            step=1,
            label="Number of content control layer",
            info="higher values make it more similar to original image. Default to control first 4 layers",
        )
        style_slider = gr.Slider(
            0,
            9,
            value=9,
            step=1,
            label="Number of style control layer",
            info="higher values make it more similar to target style. Default to control first 9 layers, usually not necessary to change.",
        )
        seed_slider = gr.Slider(
            0,
            100,
            value=0,
            step=1,
            label="Seed",
            info="Random seed for the model",
        )
        app = gr.Interface(
            fn=style_transfer_app,
            inputs=[
                text_input,
                image_input,
                cfg_slider,
                content_slider,
                style_slider,
                seed_slider,
            ],
            outputs=["image"],
            title="Artist Interactive Demo",
            examples=examples,
        )
        app.launch()
