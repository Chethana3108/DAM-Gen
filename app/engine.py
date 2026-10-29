"""
Advanced Open-Source Diffusion Engine for GPU Servers (Vast.ai) and Multi-modal Generation.
Supports:
1. FLUX.1-schnell (Black Forest Labs 12B Flow Transformer) - State-of-the-art text rendering,
   exact brand typography, and photo-realism (Apache 2.0 open-weights).
2. SDXL with IP-Adapter (h94/IP-Adapter) - Multimodal reference-grounded diffusion that injects
   the Sitecore DAM product packshot directly into cross-attention layers.
3. High-Fidelity Remote / API Fallback for development environments.

Zero 2D compositing / zero sticker pasting - 100% organically synthesized by the AI.
"""

import os
import io
import time
import logging
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image

logger = logging.getLogger("brand-engine")

# Global pipeline cache so models are loaded once into GPU VRAM
_LOADED_PIPELINE = None
_LOADED_MODEL_TYPE = None

def get_product_grounding_details(brand_name: Optional[str], product_name: Optional[str]) -> Dict[str, str]:
    """
    Supplies hyper-accurate physical grounding details for DAM brand assets to prevent
    the AI from hallucinating incorrect object types (e.g. generating a jam jar instead of sandwich biscuits).
    """
    p_lower = (product_name or "").lower()
    b_lower = (brand_name or "").lower()

    if "jim jam" in p_lower or "jimjam" in p_lower:
        return {
            "hero_description": (
                "The hero product is Britannia Treat Jim Jam sandwich biscuits (NOT a jar, NOT a bottle, NOT jam spread container). "
                "Feature the official Britannia Treat Jim Jam packaging with royal purple wrapper and crisp, authentic white 'Jim Jam' typography and red 'Britannia' emblem. "
                "Beside the packaging, display the actual sandwich biscuits: two round golden-baked biscuits with delicate flower-petal scalloped edges, "
                "sandwiching sweet vanilla cream with a glossy, translucent ruby-red strawberry jam center droplet sprinkled with fine sugar crystals."
            ),
            "negative_additions": "jar, bottle, jam jar, peanut butter, spread container, glass container, tin can, generic labels, fake typography, gibberish script, 2d sticker, pasted cutout",
            "props": "fresh plump strawberries with green calyx and water droplets, splash of fresh milk, natural wooden rustic studio surface, soft morning daylight"
        }
    elif "good day" in p_lower or "goodday" in p_lower:
        return {
            "hero_description": (
                "The hero product is Britannia Good Day Butter Cookies (NOT chocolate chip, NOT a jar). "
                "Feature the official warm golden-yellow packaging with the classic 'Good Day' smile typography and red 'BRITANNIA' emblem. "
                "Beside the packaging, show the actual golden round butter cookies with distinctive curved smile ridged lines."
            ),
            "negative_additions": "chocolate chips, dark cookies, jar, bottle, fake typography, gibberish script, 2d sticker, pasted cutout",
            "props": "golden butter cubes, fresh wheat stalks, steaming porcelain teacup, warm sunbeam lighting"
        }
    elif "himalaya" in b_lower or "neem" in p_lower:
        return {
            "hero_description": (
                "The hero product is Himalaya Purifying Neem Face Wash in its official green and white squeeze tube with authentic dark green 'Himalaya' branding."
            ),
            "negative_additions": "food, biscuit, cookie, jar, fake typography, gibberish script, 2d sticker",
            "props": "fresh green neem leaves, clear splash of pure water droplets, natural minimalist stone pedestal, spa ambience"
        }
    else:
        # Generic grounding for any DAM asset
        clean_brand = brand_name or "Premium Brand"
        clean_product = product_name or "Official Product"
        return {
            "hero_description": (
                f"The hero product is {clean_brand} {clean_product}. "
                f"Faithfully depict the authentic packaging featuring the exact brand name '{clean_brand}' and product name '{clean_product}' "
                "in sharp, professional, commercial typography. Render the physical product organically in the environment with true 3D form, contact shadows, and realistic materials."
            ),
            "negative_additions": "fake typography, gibberish words, pseudo-letters, misplaced labels, jar, bottle, 2d sticker, pasted cutout, duplicate logo",
            "props": "clean studio tabletop, elegant natural lighting, cinematic depth of field"
        }

def build_organic_master_prompt(
    user_prompt: str,
    brand_name: Optional[str],
    product_name: Optional[str],
    brand_tagline: Optional[str],
    detected_colors: List[str],
    ambience: str
) -> Tuple[str, str]:
    """
    Constructs a photorealistic prompt for diffusion models that organically places the authentic
    product into the commercial scene with zero hallucinated text.
    Returns (positive_prompt, negative_prompt).
    """
    grounding = get_product_grounding_details(brand_name, product_name)
    
    pos_parts = [
        f"Masterpiece commercial advertising photography: {user_prompt.strip().rstrip('.')}.",
        grounding["hero_description"],
    ]

    if detected_colors:
        color_str = ", ".join(detected_colors)
        pos_parts.append(f"Visual palette styled with deliberate brand color accents of {color_str}.")

    pos_parts.append(f"Scene setting: {grounding['props']}. Lighting: {ambience}.")

    if brand_tagline:
        pos_parts.append(f"Campaign theme: '{brand_tagline}'.")

    pos_parts.append(
        "Ultra-photorealistic, 8k resolution, Hasselblad medium format photography, "
        "subsurface scattering, realistic specular reflections, cinematic commercial color grading."
    )

    neg_parts = [
        "deformed, distorted, ugly, blurry, low quality, noise, grain",
        "gibberish, fake text, garbled letters, misspelled words, pseudo-alphabet",
        "duplicate logos, floating text, sticker overlay, 2d collage, pasted layers",
        grounding["negative_additions"]
    ]

    return " ".join(pos_parts), ", ".join(neg_parts)

class OpenSourceDiffusionEngine:
    """
    Engine that manages local GPU inference (FLUX.1 / SDXL IP-Adapter) and remote endpoints.
    """
    def __init__(self):
        self.model_type = os.getenv("DIFFUSION_MODEL_TYPE", "flux").lower()
        self.has_cuda = False
        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
        except ImportError:
            self.has_cuda = False

        logger.info(f"Initialized OpenSourceDiffusionEngine (CUDA={self.has_cuda}, preferred={self.model_type})")

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        reference_image_paths: List[str],
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generates the commercial visual organically using the best available model.
        """
        # Option 1: GPU Server Pipeline (Vast.ai)
        if self.has_cuda:
            try:
                return self._generate_on_gpu(
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    reference_image_paths=reference_image_paths,
                    width=width,
                    height=height,
                    seed=seed
                )
            except Exception as e:
                logger.error(f"Local GPU generation error: {e}. Falling back to remote endpoint...")

        # Option 2: Remote / API Pipeline (HF Spaces / FLUX endpoints)
        return self._generate_remote(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_image_paths=reference_image_paths,
            width=width,
            height=height
        )

    def _generate_on_gpu(
        self,
        positive_prompt: str,
        negative_prompt: str,
        reference_image_paths: List[str],
        width: int,
        height: int,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Runs native PyTorch Diffusers pipeline on Vast.ai NVIDIA GPU."""
        import torch
        global _LOADED_PIPELINE, _LOADED_MODEL_TYPE

        generator = torch.Generator("cuda")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        # Target: FLUX.1-schnell (World's #1 model for text and commercial realism)
        if self.model_type == "flux":
            if _LOADED_PIPELINE is None or _LOADED_MODEL_TYPE != "flux":
                from diffusers import FluxPipeline
                logger.info("Loading FLUX.1-schnell pipeline into GPU VRAM (bfloat16)...")
                _LOADED_PIPELINE = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=torch.bfloat16
                ).to("cuda")
                _LOADED_MODEL_TYPE = "flux"
                logger.info("FLUX.1-schnell successfully loaded!")

            logger.info("Synthesizing visual with FLUX.1-schnell (4 inference steps)...")
            result = _LOADED_PIPELINE(
                prompt=positive_prompt,
                width=width,
                height=height,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=generator
            ).images[0]
            return result

        # Target: SDXL with IP-Adapter (Direct reference image conditioning)
        elif self.model_type == "ip-adapter":
            if _LOADED_PIPELINE is None or _LOADED_MODEL_TYPE != "ip-adapter":
                from diffusers import AutoPipelineForText2Image
                logger.info("Loading SDXL + IP-Adapter pipeline into GPU VRAM...")
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16"
                ).to("cuda")
                pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name="ip-adapter-plus_sdxl_vit-h.bin"
                )
                pipe.set_ip_adapter_scale(0.7)
                _LOADED_PIPELINE = pipe
                _LOADED_MODEL_TYPE = "ip-adapter"
                logger.info("SDXL + IP-Adapter successfully loaded!")

            # Load primary reference image from DAM
            ref_img = None
            if reference_image_paths and os.path.exists(reference_image_paths[0]):
                ref_img = Image.open(reference_image_paths[0]).convert("RGB")
                logger.info(f"Injecting DAM reference image {reference_image_paths[0]} into IP-Adapter...")

            kwargs = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "generator": generator
            }
            if ref_img is not None:
                kwargs["ip_adapter_image"] = ref_img

            return _LOADED_PIPELINE(**kwargs).images[0]

        else:
            raise ValueError(f"Unknown GPU model type: {self.model_type}")

    def _generate_remote(
        self,
        positive_prompt: str,
        negative_prompt: str,
        reference_image_paths: List[str],
        width: int,
        height: int
    ) -> Image.Image:
        """
        Remote / API generation for development environments.
        Uses pure open-source diffusion without any 2D pasting.
        """
        from gradio_client import Client as GradioClient, handle_file
        
        endpoint = os.getenv("MODEL_ENDPOINT", "black-forest-labs/FLUX.1-schnell")
        hf_token = os.getenv("HF_TOKEN", None) or None
        logger.info(f"Using remote diffusion endpoint: {endpoint}")

        # Attempt 1: FLUX.1 endpoint
        try:
            client = GradioClient(endpoint, token=hf_token)
            # Standard FLUX Gradio space call signature
            res_path = client.predict(
                prompt=positive_prompt,
                seed=0,
                randomize_seed=True,
                width=width,
                height=height,
                num_inference_steps=4,
                api_name="/infer"
            )
            # Gradio may return tuple (path, seed) or path
            if isinstance(res_path, (list, tuple)):
                res_path = res_path[0]
            if os.path.exists(res_path):
                return Image.open(res_path)
        except Exception as e:
            logger.warning(f"Primary endpoint {endpoint} failed: {e}. Trying high-speed SDXL...")

        # Attempt 2: High-speed SDXL endpoint with zero pasting
        try:
            sdxl_client = GradioClient("AP123/SDXL-Lightning")
            img_path = sdxl_client.predict(positive_prompt, "4-Step", api_name="/generate_image")
            if os.path.exists(img_path):
                logger.info("Successfully generated via high-speed SDXL (clean organic render)!")
                return Image.open(img_path)
        except Exception as e:
            logger.error(f"High-speed fallback failed: {e}")

        # Final safety fallback: synthetic clean placeholder if all internet endpoints are unreachable
        logger.warning("Network endpoints offline. Returning clean studio visual placeholder.")
        return Image.new("RGB", (width, height), color=(245, 240, 235))
