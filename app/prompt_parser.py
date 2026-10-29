"""
Natural-Language Creative Prompt Understanding Module.
Parses user instructions to dynamically extract:
1. Target asset references (e.g. Britannia_JimJam.png, filename, or product name).
2. Requested environment, setting, and background (beach, wooden table, modern kitchen, luxury marble, etc.).
3. Requested interaction mode (tabletop, handheld/person, lifestyle, action).
4. Co-objects (coffee cup, tea, fruits, kitchen utensils, etc.).
5. Visual style, lighting, and color harmony.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("prompt-parser")

@dataclass
class ParsedPrompt:
    """Structured understanding of the user's natural language creative instruction."""
    raw_prompt: str
    target_asset_filename: Optional[str] = None
    target_asset_query: Optional[str] = None
    interaction_mode: str = "tabletop"  # tabletop, handheld_person, lifestyle_setting, dynamic_action
    scene_environment: str = "commercial advertising studio"
    co_objects: List[str] = field(default_factory=list)
    lighting_style: str = "soft commercial studio lighting"
    composition_placement: str = "center"  # center, lower_left, lower_right, hand_grip
    aspect_ratio: str = "1:1"
    preserve_product_strictly: bool = True

class PromptParser:
    """Parses natural-language creative prompts without any hardcoded brand logic."""

    # Common image file extensions to detect explicit filenames in prompts
    FILE_EXT_PATTERN = re.compile(r'\b([a-zA-Z0-9_\-\.]+\.(?:png|jpg|jpeg|webp|gif))\b', re.IGNORECASE)

    @classmethod
    def parse(cls, prompt: str) -> ParsedPrompt:
        """Parses the user prompt into structured creative attributes."""
        clean_prompt = prompt.strip()
        p_lower = clean_prompt.lower()

        # 1. Detect explicit asset filename in prompt (e.g. "Use Britannia_JimJam.png on a wooden table...")
        target_filename = None
        target_query = None
        match = cls.FILE_EXT_PATTERN.search(clean_prompt)
        if match:
            target_filename = match.group(1)
            target_query = target_filename
            logger.info(f"Detected explicit asset filename in prompt: '{target_filename}'")
        else:
            # Check for phrases like "this jimjam biscuit", "the good day pack", "this product"
            prod_match = re.search(r'(?:this|the|for|with)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+packet|\s+biscuit|\s+product|\s+is|\s+on|\s+in|\s+to|\s+being)', p_lower)
            if prod_match:
                extracted = prod_match.group(1).strip()
                if len(extracted) > 2 and extracted not in ["product", "asset", "image", "item"]:
                    target_query = extracted

        # 2. Detect Interaction Mode
        is_held = any(w in p_lower for w in ["hold", "holding", "in hand", "in hands", "held by", "gripping", "clutching"])
        is_person = any(w in p_lower for w in ["person", "woman", "man", "model", "celebrity", "actor", "chef", "barista", "hands"])
        is_action = any(w in p_lower for w in ["splash", "splashing", "pouring", "floating", "flying", "burst", "motion", "running"])
        is_tabletop = any(w in p_lower for w in ["table", "desk", "counter", "surface", "marble", "wood", "tray", "shelf", "pedestal", "placed on", "resting on", "sitting on"])

        if is_held or is_person:
            interaction_mode = "handheld_person"
            composition = "hand_grip"
        elif is_action:
            interaction_mode = "dynamic_action"
            composition = "center"
        elif is_tabletop:
            interaction_mode = "tabletop"
            composition = "center"
        else:
            interaction_mode = "lifestyle_setting"
            composition = "center"

        # 3. Detect Co-Objects Requested
        co_objects = []
        if any(w in p_lower for w in ["coffee", "latte", "cappuccino", "espresso"]):
            co_objects.append("a steaming white ceramic cup of artisanal coffee with latte art")
        elif any(w in p_lower for w in ["tea", "chai", "teacup"]):
            co_objects.append("a steaming fine porcelain teacup with gentle steam")

        if any(w in p_lower for w in ["strawberry", "strawberries", "berry", "berries"]):
            co_objects.append("fresh plump strawberries with water droplets")

        if any(w in p_lower for w in ["milk", "cream", "splash"]):
            co_objects.append("clean dynamic milk splash droplets")

        if any(w in p_lower for w in ["flower", "leaves", "botanical", "herbal", "neem"]):
            co_objects.append("fresh natural botanical green leaves with dew")

        # 4. Detect Scene Environment
        if "beach" in p_lower:
            environment = "a serene sunlit tropical beach with soft golden sand and ocean waves in the distance"
            lighting = "bright natural coastal daylight with warm golden reflections"
        elif any(w in p_lower for w in ["kitchen", "modern kitchen"]):
            environment = "a sleek, sunlit modern kitchen counter with clean contemporary marble and minimalist styling"
            lighting = "clean natural daylight streaming through large kitchen windows"
        elif any(w in p_lower for w in ["cafe", "coffee shop"]):
            environment = "a cozy warm artisanal cafe interior with soft ambient bokeh"
            lighting = "warm golden morning sunlight with gentle directional rays and soft luminous ambient fill"
        elif any(w in p_lower for w in ["wooden table", "wood table", "rustic"]):
            environment = "a rustic reclaimed wooden tabletop with rich authentic grain texture"
            lighting = "warm morning sunlight casting gentle diagonal rays across the wooden surface"
        elif any(w in p_lower for w in ["luxury", "premium", "marble"]):
            environment = "a high-end luxury marble pedestal in an elegant minimalist architectural studio"
            lighting = "dramatic high-contrast commercial lighting with specular highlights and soft rim light"
        elif any(w in p_lower for w in ["outdoor", "lifestyle", "garden", "park"]):
            environment = "an inviting outdoor lifestyle setting with lush greenery and soft natural atmosphere"
            lighting = "bright crisp outdoor natural daylight with dappled organic shadows"
        else:
            environment = "a professional commercial advertising studio surface with spacious foreground"
            lighting = "balanced commercial studio lighting with soft diffused highlights"

        # If coffee or tea is on the table, shift product to lower left/center so co-object has space
        if co_objects and interaction_mode == "tabletop":
            composition = "lower_left_balanced"

        return ParsedPrompt(
            raw_prompt=clean_prompt,
            target_asset_filename=target_filename,
            target_asset_query=target_query,
            interaction_mode=interaction_mode,
            scene_environment=environment,
            co_objects=co_objects,
            lighting_style=lighting,
            composition_placement=composition,
            preserve_product_strictly=True
        )

    @classmethod
    def build_environment_generation_prompt(cls, parsed: ParsedPrompt) -> Tuple[str, str]:
        """
        Synthesizes a dedicated prompt for the AI to generate ONLY the surrounding environment and props,
        guaranteeing that the product itself is NOT drawn by AI, preventing fake typos and hallucinations.
        """
        prop_str = f"Featuring {', '.join(parsed.co_objects)} on the right side." if parsed.co_objects else "Minimalist professional commercial composition."

        pos_parts = [
            "Masterpiece commercial advertising scene photography.",
            f"Environment: {parsed.scene_environment}.",
            f"Props & arrangement: {prop_str}",
            f"Lighting: {parsed.lighting_style}.",
            "Empty spacious tabletop surface in the lower center foreground ready for commercial hero product placement.",
            "Shot on Hasselblad H6D-100c, 8k resolution, razor-sharp focus on foreground surface, beautiful optical depth of field bokeh in background, photorealistic color grading."
        ]

        neg_parts = [
            "deformed, blurry, ugly, low quality, noise, grain, cartoon, illustration, drawing",
            "fake packaging, boxes, biscuits, jars, bottles, cans, letters, typography, words, text, labels, logos in foreground",
            "watermark, signature, bad proportions, border, frame"
        ]

        return " ".join(pos_parts), ", ".join(neg_parts)
