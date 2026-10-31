"""
Clean, Production-Grade GPT Image Engine.
Powered by OpenAI GPT Image model (e.g. gpt-image-2.5-sunburst)
for reference-guided commercial advertising visual generation, editing, and variations.
"""

import os
import logging
from typing import List, Optional, Union
from PIL import Image

from app.gpt_image_service import GPTImageService

logger = logging.getLogger("gpt-engine")

class GPTImageEngine:
    """
    Core image generation engine wrapping GPT Image service.
    Directly routes reference assets to the image model's editing/reference endpoints.
    """
    def __init__(self):
        self.model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")
        self.gpt_service = GPTImageService(model=self.model)
        logger.info("Initialized GPTImageEngine with model '%s'", self.model)

    @property
    def is_available(self) -> bool:
        return self.gpt_service.is_available

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        reference_image_paths: Optional[List[str]] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        mask_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generates commercial visual scenes. If reference image paths are supplied,
        they are GENUINELY forwarded to the reference-guided generation / editing API.
        """
        size_str = f"{width}x{height}"

        if reference_image_paths and len(reference_image_paths) > 0:
            primary_ref = reference_image_paths[0]
            additional_refs = reference_image_paths[1:] if len(reference_image_paths) > 1 else None
            logger.info("Generating image using authentic reference asset: %s", primary_ref)
            return self.gpt_service.generate_with_reference(
                prompt=positive_prompt,
                reference_image=primary_ref,
                additional_references=additional_refs,
                mask=mask_path,
                size=size_str
            )
        else:
            logger.info("Generating image using text-to-image prompt: %s...", positive_prompt[:80])
            return self.gpt_service.generate(prompt=positive_prompt, size=size_str)

    def edit(
        self,
        image: Union[Image.Image, str, bytes],
        prompt: str,
        mask: Optional[Union[Image.Image, str, bytes]] = None,
        width: int = 1024,
        height: int = 1024
    ) -> Image.Image:
        """
        Applies a targeted edit or refinement to an existing image.
        """
        size_str = f"{width}x{height}"
        return self.gpt_service.edit(
            image=image,
            prompt=prompt,
            size=size_str,
            mask=mask
        )

    def refine(
        self,
        image: Union[Image.Image, str, bytes],
        refinement_instruction: str,
        original_prompt: str,
        width: int = 1024,
        height: int = 1024
    ) -> Image.Image:
        """
        Refines a generated visual to correct specific QA defects.
        """
        targeted_prompt = (
            f"Modify this commercial image to fix the following issues: {refinement_instruction}. "
            f"Original requirement: {original_prompt}. "
            "Maintain professional advertising composition, photorealistic textures, and seamless integration."
        )
        return self.edit(
            image=image,
            prompt=targeted_prompt,
            width=width,
            height=height
        )

    def generate_variation(
        self,
        prompt: str,
        reference_image_path: Optional[str] = None,
        previous_image: Optional[Image.Image] = None,
        width: int = 1024,
        height: int = 1024,
        variation_index: int = 1
    ) -> Image.Image:
        """
        Generates aesthetic variations using the reference asset or previous composition.
        """
        return self.gpt_service.generate_variation(
            prompt=prompt,
            reference_image=reference_image_path,
            previous_image=previous_image,
            size=f"{width}x{height}",
            variation_index=variation_index
        )

# Backward-compatible alias
OpenSourceDiffusionEngine = GPTImageEngine
