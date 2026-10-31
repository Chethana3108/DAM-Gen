"""
Dynamic Natural-Language Creative Understanding Module.
Replaces brittle keyword matching and brand hardcoding with dynamic,
AI-driven multimodal prompt and asset understanding.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("prompt-parser")

@dataclass
class ParsedPrompt:
    """Dynamic, AI-driven understanding of the user's natural language creative instruction."""
    raw_prompt: str
    target_asset_filename: Optional[str] = None
    target_asset_query: Optional[str] = None
    subject_description: Optional[str] = None
    scene_environment: str = ""
    composition_guidance: str = ""
    elements_to_preserve: str = ""
    unified_generation_prompt: str = ""
    aspect_ratio: str = "1:1"
    ai_insights: Dict[str, Any] = field(default_factory=dict)

    # Backward-compatible convenience properties
    @property
    def interaction_mode(self) -> str:
        return self.ai_insights.get("interaction_mode", "dynamic")

    @property
    def co_objects(self) -> List[str]:
        return self.ai_insights.get("co_objects", [])

    @property
    def lighting_style(self) -> str:
        return self.ai_insights.get("lighting_style", "natural commercial lighting")

    @property
    def composition_placement(self) -> str:
        return self.ai_insights.get("composition_placement", "contextual")

class PromptParser:
    """
    Understands creative intent dynamically without hardcoded brand lists,
    fixed scene classifications, or predetermined composition rules.
    """

    FILE_EXT_PATTERN = re.compile(r'\b([a-zA-Z0-9_\-\.]+\.(?:png|jpg|jpeg|webp|gif))\b', re.IGNORECASE)

    @classmethod
    def parse(
        cls,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        reference_image: Optional[Any] = None,
        gpt_service: Optional[Any] = None
    ) -> ParsedPrompt:
        """
        Parses creative instructions. If a gpt_service and reference_image are provided,
        utilizes multimodal AI understanding. Otherwise uses dynamic semantic extraction.
        """
        clean_prompt = prompt.strip()
        metadata = metadata or {}

        # 1. Detect explicit asset filename in prompt if specified by user
        target_filename = None
        target_query = None
        match = cls.FILE_EXT_PATTERN.search(clean_prompt)
        if match:
            target_filename = match.group(1)
            target_query = target_filename
            logger.info("Detected explicit asset filename in prompt: '%s'", target_filename)
        else:
            # Extract natural subject query without hardcoded brand keywords
            prod_match = re.search(
                r'(?:this|the|for|with)\s+([a-zA-Z0-9_\-\s]{3,35}?)(?:\s+packet|\s+bottle|\s+pack|\s+product|\s+is|\s+on|\s+in|\s+to|\s+beside|\s+near|\s+with)',
                clean_prompt,
                re.IGNORECASE
            )
            if prod_match:
                candidate = prod_match.group(1).strip()
                if candidate.lower() not in ["product", "asset", "image", "item", "picture"]:
                    target_query = candidate

        # 2. Multimodal AI-driven intent analysis if service and image available
        ai_insights: Dict[str, Any] = {}
        if gpt_service is not None and reference_image is not None and getattr(gpt_service, "is_available", False):
            try:
                ai_insights = gpt_service.analyze_creative_intent(
                    prompt=clean_prompt,
                    reference_image=reference_image,
                    metadata=metadata
                )
            except Exception as e:
                logger.warning("Error running AI multimodal creative analysis (%s); using dynamic fallback.", e)

        # 3. Build dynamic unified prompt
        subject_title = metadata.get("product_name") or metadata.get("title") or target_query or "product"
        
        if ai_insights.get("generation_prompt"):
            unified_prompt = ai_insights["generation_prompt"]
            subject_desc = ai_insights.get("subject_description", subject_title)
            scene_env = ai_insights.get("user_environment", clean_prompt)
            comp_guide = ai_insights.get("composition_and_lighting", "natural commercial composition")
            preserve_guide = ai_insights.get("elements_to_preserve", "branding and packaging geometry")
        else:
            subject_desc = f"authentic {subject_title} from the reference image"
            scene_env = clean_prompt
            comp_guide = "flawless commercial advertising lighting, realistic perspective, and authentic contact shadows"
            preserve_guide = "exact packaging graphics, typography, labels, and product geometry"
            unified_prompt = (
                f"Commercial advertising visual featuring the {subject_desc}, "
                f"composed naturally within the requested setting: {clean_prompt}. "
                f"Preserve {preserve_guide}. "
                f"Lighting & composition: {comp_guide}. "
                "Shot with high-end commercial advertising photography aesthetics, sharp foreground detail, and realistic environmental integration."
            )

        return ParsedPrompt(
            raw_prompt=clean_prompt,
            target_asset_filename=target_filename,
            target_asset_query=target_query,
            subject_description=subject_desc,
            scene_environment=scene_env,
            composition_guidance=comp_guide,
            elements_to_preserve=preserve_guide,
            unified_generation_prompt=unified_prompt,
            ai_insights=ai_insights
        )

    @classmethod
    def build_environment_generation_prompt(cls, parsed: ParsedPrompt) -> Tuple[str, str]:
        """
        Maintains backward compatibility while pointing to the unified generation prompt.
        """
        pos_prompt = parsed.unified_generation_prompt or (
            f"Commercial advertising photography: {parsed.raw_prompt}. "
            "High resolution, sharp focus, professional studio lighting, realistic depth and textures."
        )
        neg_prompt = (
            "blurry, distorted, low quality, deformed, artificial sticker, cut-and-paste collage, "
            "crude borders, floating ungrounded objects, watermark"
        )
        return pos_prompt, neg_prompt
