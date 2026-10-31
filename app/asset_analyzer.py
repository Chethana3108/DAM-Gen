"""
Intelligent Source-Asset Understanding Module.
Analyzes Sitecore Content Hub DAM assets dynamically:
1. Identifies asset type: transparent PNG cutout, layered composition, studio packshot, or photograph.
2. Preserves authentic source imagery without destructive edge stripping or arbitrary component clipping.
3. Never modifies the original DAM asset.
4. Supports linked transparent layers and optional non-destructive segmentation.
"""

import os
import io
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from PIL import Image, ImageStat
import numpy as np

logger = logging.getLogger("asset-analyzer")

@dataclass
class AnalyzedAsset:
    """Analytical properties of the authentic DAM asset."""
    file_path: str
    original_size: Tuple[int, int]
    has_alpha: bool
    asset_type: str  # "transparent_cutout", "layered_composition", "studio_packshot", "full_photograph"
    original_image: Image.Image
    cutout_image: Image.Image
    cutout_bbox: Tuple[int, int, int, int]
    cutout_size: Tuple[int, int]
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    average_luminance: float = 0.5
    object_category: str = "product"
    linked_layer_paths: List[str] = field(default_factory=list)
    segmentation_applied: bool = False

class AssetAnalyzer:
    """
    Analyzes DAM assets intelligently. Does NOT automatically run background removal
    or mutilate pixels along image edges. Respects authentic asset integrity.
    """

    @classmethod
    def analyze(
        cls,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        linked_layer_paths: Optional[List[str]] = None,
        require_cutout_extraction: bool = False
    ) -> AnalyzedAsset:
        """
        Analyzes an asset from Sitecore DAM.
        Keeps original image fully intact. Only extracts cutout if explicitly required.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Asset file not found: {file_path}")

        raw_img = Image.open(file_path)
        orig_size = raw_img.size
        metadata = metadata or {}
        linked_layer_paths = linked_layer_paths or []

        # 1. Determine if asset contains an authentic transparent alpha channel
        has_alpha = False
        cutout = None
        asset_type = "full_photograph"

        if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
            rgba = raw_img.convert("RGBA")
            alpha_extrema = rgba.split()[3].getextrema()
            if alpha_extrema[0] < 240:
                has_alpha = True
                asset_type = "transparent_cutout"
                bbox = rgba.getbbox()
                cutout = rgba.crop(bbox) if bbox else rgba
                logger.info("Asset '%s' contains an authentic transparent alpha layer.", os.path.basename(file_path))

        # 2. Check if there are linked layers with transparency
        if not has_alpha and linked_layer_paths:
            for lpath in linked_layer_paths:
                if os.path.exists(lpath):
                    try:
                        l_img = Image.open(lpath)
                        if l_img.mode in ("RGBA", "LA"):
                            l_alpha = l_img.split()[3].getextrema()
                            if l_alpha[0] < 240:
                                asset_type = "layered_composition"
                                logger.info("Found authentic transparent linked layer: %s", os.path.basename(lpath))
                                break
                    except Exception:
                        pass

        # 3. For opaque assets, check if studio packshot or photographic scene
        if not cutout:
            if cls._is_studio_neutral_background(raw_img):
                asset_type = "studio_packshot"

            if require_cutout_extraction:
                # Segment only if specifically required (e.g. for compositing fallback)
                cutout = cls._safe_segmentation(raw_img, file_path)
            else:
                # Default: product visual is the full image, no destructive edge cropping
                cutout = raw_img.convert("RGBA")

        bbox = cutout.getbbox() or (0, 0, cutout.width, cutout.height)
        cutout_cropped = cutout.crop(bbox)

        # 4. Color and luminance analysis
        rgb_img = raw_img.convert("RGB")
        stat = ImageStat.Stat(rgb_img)
        avg_lum = sum(stat.mean[:3]) / (3.0 * 255.0)
        colors = cls._get_dominant_colors(raw_img)

        # 5. Extract category from metadata dynamically
        cat = metadata.get("product_category") or metadata.get("content_type") or "commercial_product"

        return AnalyzedAsset(
            file_path=file_path,
            original_size=orig_size,
            has_alpha=has_alpha,
            asset_type=asset_type,
            original_image=raw_img,
            cutout_image=cutout_cropped,
            cutout_bbox=bbox,
            cutout_size=cutout_cropped.size,
            dominant_colors=colors,
            average_luminance=avg_lum,
            object_category=cat,
            linked_layer_paths=linked_layer_paths,
            segmentation_applied=require_cutout_extraction
        )

    @classmethod
    def _is_studio_neutral_background(cls, img: Image.Image) -> bool:
        """Determines if the image has a uniform studio packshot background."""
        w, h = img.size
        rgb = img.convert("RGB")
        corners = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((w - 1, 0)),
            rgb.getpixel((0, h - 1)),
            rgb.getpixel((w - 1, h - 1))
        ]
        # Check corner consistency
        c0 = np.array(corners[0])
        dists = [np.linalg.norm(np.array(c) - c0) for c in corners[1:]]
        return all(d < 30 for d in dists)

    @classmethod
    def _safe_segmentation(cls, img: Image.Image, file_path: str) -> Image.Image:
        """
        Applies neural segmentation safely without arbitrarily zeroing out edge borders
        or destroying valid product packaging.
        """
        try:
            import rembg
            logger.info("Applying non-destructive rembg segmentation on: %s", os.path.basename(file_path))
            return rembg.remove(img.convert("RGBA"))
        except Exception as e:
            logger.warning("Segmentation unavailable or failed (%s); returning intact image.", e)
            return img.convert("RGBA")

    @classmethod
    def _get_dominant_colors(cls, img: Image.Image, num_colors: int = 4) -> List[Tuple[int, int, int]]:
        """Extracts dominant RGB colors from the asset."""
        small = img.convert("RGB").resize((100, 100), Image.Resampling.BOX)
        arr = np.array(small)
        rgb = arr.reshape(-1, 3)
        # Quantize colors
        quantized = (rgb // 32) * 32
        unique, counts = np.unique(quantized, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[::-1][:num_colors]
        return [tuple(int(c) for c in unique[idx]) for idx in top_indices]
