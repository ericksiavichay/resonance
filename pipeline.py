import torch
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)


def generate(
    prompt,
    negative_prompt,
    input_image,
    strength,
    n_steps,
    cfg=None,
    scheduler=None,
    models={},
    seed=None,
    device="cuda",
):
    # inference from the model
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be in (0, 1]")

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generate.manual_seed(seed)

        clap = models["clap"].to(device)

        if cfg is not None:
            prompt_embeddings = clap.text_encoder(prompt)
            negative_prompt_embeddings = clap.text_encoder(negative_prompt)
            context = torch.cat([prompt_embeddings, negative_prompt_embeddings])
        else:
            context = clap.text_encoder(prompt)
        clap.to("cpu")

        # scheduler
        scheduler = DDPMScheduler(n_steps)

        if input_image:
            image_encoder = models["image_encoder"].to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device
            )
            latents = encoder(input_image_tensor, encoder_noise)

            # TODO: set strength and add noise to the latents according to the strength
            start_step = n_steps - int(n_steps * strength)
            scheduler.timesteps = scheduler.timesteps[start_step:]
            latents = scheduler.add_noise(
                latents,
                noise=torch.randn(
                    latents.shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                ),
                timesteps=scheduler.timesteps[0],
            )

        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        unet = models["unet"].to(device)
        timesteps = tqdm(scheduler.timesteps)
        for i, timestep in enumerate(timesteps):
            model_input = latents

            if cfg is not None:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = unet(model_input, timestep, context)

            if cfg is not None:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg * (output_cond - output_uncond) * output_uncond

            # noise removal
            latents, _ = scheduler.step(
                model_output, timestep, latents, return_dict=False
            )

        unet.to("cpu")

        image_decoder = models["image_decoder"].to(device)

        images = decoder(latents)
        image_decoder.to("cpu")

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images
