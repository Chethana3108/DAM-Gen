"""
Asset Understanding & Object Extraction Module.
Analyzes Sitecore Content Hub DAM assets:
1. Detects transparent PNG layers vs complete photographic assets.
2. Extracts clean product cutouts using intelligent segmentation where required.
3. Analyzes object dimensions, bounding boxes, dominant color palettes, and visual features.
"""

import os
import io
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from PIL import Image, ImageFilter, ImageOps, ImageStat
import numpy as np

logger = logging.getLogger("asset-analyzer")

# In-memory cache for extracted cutouts
_CUTOUT_CACHE: Dict[str, Image.Image] = {}

@dataclass
class AnalyzedAsset:
    """Holds analytical properties of the DAM asset."""
    file_path: str
    original_size: Tuple[int, int]
    has_alpha: bool
    cutout_image: Image.Image
    cutout_bbox: Tuple[int, int, int, int]
    cutout_size: Tuple[int, int]
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    average_luminance: float = 0.5
    object_category: str = "product"

class AssetAnalyzer:
    """
    Understands DAM assets and extracts products with pixel-level precision.
    Does NOT assume every asset is a cutout; handles transparent layers,
    studio packshots, and full photographs.
    """

    @classmethod
    def analyze(cls, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> AnalyzedAsset:
        """Analyzes an image file, extracts the clean product cutout, and computes visual features."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Asset file not found: {file_path}")

        raw_img = Image.open(file_path)
        orig_size = raw_img.size

        # 1. Determine if the asset already has an authentic transparent alpha channel
        has_alpha = False
        cutout = None

        if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
            rgba = raw_img.convert("RGBA")
            alpha_extrema = rgba.split()[3].getextrema()
            # If there are authentic transparent pixels (min alpha < 240)
            if alpha_extrema[0] < 240:
                has_alpha = True
                bbox = rgba.getbbox()
                cutout = rgba.crop(bbox) if bbox else rgba
                logger.info(f"Asset '{os.path.basename(file_path)}' already contains an authentic transparent alpha layer.")

        # 2. If photographic / opaque background, extract object via neural segmentation (rembg)
        # and isolate the true hero product packaging, stripping away any cut-off frame props (wire racks, partial dishes)
        if not cutout:
            segmented = cls._extract_segmentation(raw_img, file_path)
            cutout = cls._isolate_hero_product(segmented)

        bbox = cutout.getbbox() or (0, 0, cutout.width, cutout.height)
        cutout_cropped = cutout.crop(bbox)

        # 3. Compute dominant colors & luminance
        rgb_img = cutout_cropped.convert("RGB")
        stat = ImageStat.Stat(rgb_img)
        avg_lum = sum(stat.mean[:3]) / (3.0 * 255.0)

        # Extract top dominant colors from non-transparent pixels
        colors = cls._get_dominant_colors(cutout_cropped)

        # 4. Infer object category from metadata if available
        cat = "product"
        if metadata:
            text_corpus = f"{metadata.get('title', '')} {metadata.get('filename', '')} {metadata.get('alt_text', '')}".lower()
            if any(w in text_corpus for w in ["biscuit", "cookie", "cracker", "snack", "food"]):
                cat = "food_biscuit"
            elif any(w in text_corpus for w in ["bottle", "jar", "tube", "cosmetic", "wash", "lotion", "cream"]):
                cat = "bottle_or_tube"
            elif any(w in text_corpus for w in ["can", "beverage", "soda", "drink"]):
                cat = "beverage_can"
            elif any(w in text_corpus for w in ["box", "carton", "pack"]):
                cat = "packaged_box"

        return AnalyzedAsset(
            file_path=file_path,
            original_size=orig_size,
            has_alpha=has_alpha,
            cutout_image=cutout_cropped,
            cutout_bbox=bbox,
            cutout_size=cutout_cropped.size,
            dominant_colors=colors,
            average_luminance=avg_lum,
            object_category=cat
        )

    @classmethod
    def _isolate_hero_product(cls, cutout_rgba: Image.Image) -> Image.Image:
        """
        Intelligently isolates the authentic hero product packaging from complex flat-lay or multi-object scenes.
        Strictly strips out background props that clip the frame borders (such as wire racks, cut-off dishes, or tables).
        Preserves 100% of the authentic packaging graphics, logos, and labels using convex hull polygon bounds.
        """
        import cv2

        arr = np.array(cutout_rgba)
        if arr.ndim < 3 or arr.shape[2] < 4:
            return cutout_rgba

        alpha = arr[:, :, 3]
        h, w = alpha.shape

        # Step 1: Detach edge border clipping artifacts (e.g. background wire racks or cut-off dishes in the upper/side margins)
        # We zero out only the extreme 6% border zone if it contains cut-off accessories
        clean_alpha = alpha.copy()
        clean_alpha[:int(h * 0.08), :] = 0  # top margin props
        clean_alpha[:, :int(w * 0.04)] = 0  # extreme left edge props

        # Step 2: Find connected components in the product region
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((clean_alpha > 50).astype(np.uint8))
        if num_labels <= 2:
            return cutout_rgba  # Single component or empty

        valid_components = []
        diag = (w**2 + h**2)**0.5

        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            if area < 3500:
                continue  # Skip tiny noise fragments

            cx, cy = centroids[i]
            dist_to_center = ((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5
            # Score favors large, centrally framed commercial objects
            score = float(area) * max(0.2, 1.0 - (dist_to_center / diag))

            valid_components.append({
                "label": i,
                "area": area,
                "score": score,
                "bbox": (x, y, cw, ch)
            })

        if not valid_components:
            return cutout_rgba

        # Select the highest-scoring hero component (the authentic product packaging)
        best = sorted(valid_components, key=lambda c: c["score"], reverse=True)[0]
        chosen_mask = (labels == best["label"]).astype(np.uint8)

        # Smooth and preserve packaging artwork (fill internal holes created by light packaging art)
        contours, _ = cv2.findContours(chosen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_c)
            hull_mask = np.zeros_like(alpha)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
            final_alpha = np.minimum(alpha, hull_mask)
        else:
            final_alpha = alpha * chosen_mask

        res_arr = arr.copy()
        res_arr[:, :, 3] = final_alpha
        res_img = Image.fromarray(res_arr)
        bbox = res_img.getbbox()
        return res_img.crop(bbox) if bbox else res_img

    @classmethod
    def _extract_segmentation(cls, img: Image.Image, file_path: str) -> Image.Image:
        """Applies neural segmentation (rembg) to isolate the subject from photographic backgrounds."""
        if file_path in _CUTOUT_CACHE:
            return _CUTOUT_CACHE[file_path]

        try:
            import rembg
            logger.info(f"Applying segmentation on photographic asset: {os.path.basename(file_path)}")
            res = rembg.remove(img.convert("RGBA"))
            _CUTOUT_CACHE[file_path] = res
            return res
        except Exception as e:
            logger.warning(f"rembg segmentation failed ({e}); falling back to color-keying cutout.")
            return cls._fast_color_key_cutout(img.convert("RGBA"))

    @classmethod
    def _fast_color_key_cutout(cls, img_rgba: Image.Image, tolerance: int = 25) -> Image.Image:
        """Fast edge-aware color keying for studio photography with neutral backgrounds."""
        w, h = img_rgba.size
        corners = [
            img_rgba.getpixel((2, 2))[:3],
            img_rgba.getpixel((w - 3, 2))[:3],
            img_rgba.getpixel((2, h - 3))[:3],
            img_rgba.getpixel((w - 3, h - 3))[:3]
        ]
        avg_bg = [sum(c[i] for c in corners) // 4 for i in range(3)]

        pixels = img_rgba.load()
        for y in range(h):
            for x in range(w):
                r, g, b, a = pixels[x, y]
                dist = ((r - avg_bg[0])**2 + (g - avg_bg[1])**2 + (b - avg_bg[2])**2)**0.5
                if dist < tolerance:
                    pixels[x, y] = (r, g, b, 0)
                elif dist < (tolerance + 15):
                    alpha = int(255 * (dist - tolerance) / 15)
                    pixels[x, y] = (r, g, b, alpha)
        return img_rgba

    @classmethod
    def _get_dominant_colors(cls, cutout_rgba: Image.Image, num_colors: int = 4) -> List[Tuple[int, int, int]]:
        """Extracts dominant RGB colors from the non-transparent product region."""
        small = cutout_rgba.resize((100, 100), Image.Resampling.BOX)
        arr = np.array(small)
        # Select pixels with alpha > 128
        opaque = arr[arr[..., 3] > 128]
        if len(opaque) == 0:
            return [(128, 128, 128)]
        rgb = opaque[:, :3]
        # Quantize colors
        quantized = (rgb // 32) * 32
        unique, counts = np.unique(quantized, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[::-1][:num_colors]
        return [tuple(int(c) for c in unique[idx]) for idx in top_indices]

