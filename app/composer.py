import io
import logging
from typing import Optional, Tuple
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw

logger = logging.getLogger(__name__)

def fast_product_cutout(img: Image.Image, bg_tolerance: int = 25) -> Image.Image:
    """
    Fast, reliable transparent cutout for studio product photography and packshots.
    Detects background color from image corners (usually pure white, near-white, or neutral grey)
    and produces a crisp alpha channel with smooth edge anti-aliasing.
    """
    img_rgba = img.convert("RGBA")
    w, h = img_rgba.size
    
    # Sample 4 corners to determine background color
    corners = [
        img_rgba.getpixel((2, 2))[:3],
        img_rgba.getpixel((w - 3, 2))[:3],
        img_rgba.getpixel((2, h - 3))[:3],
        img_rgba.getpixel((w - 3, h - 3))[:3]
    ]
    # Average corner color
    avg_r = sum(c[0] for c in corners) // 4
    avg_g = sum(c[1] for c in corners) // 4
    avg_b = sum(c[2] for c in corners) // 4
    
    pixels = img_rgba.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            # Euclidean color distance to background
            dist = ((r - avg_r)**2 + (g - avg_g)**2 + (b - avg_b)**2)**0.5
            if dist < bg_tolerance:
                pixels[x, y] = (r, g, b, 0)
            elif dist < (bg_tolerance + 15):
                # Smooth alpha falloff for clean anti-aliased edges
                alpha = int(255 * (dist - bg_tolerance) / 15)
                pixels[x, y] = (r, g, b, alpha)
                
    # Crop transparent bounding box
    bbox = img_rgba.getbbox()
    if bbox:
        img_rgba = img_rgba.crop(bbox)
        
    return img_rgba

def extract_product_cutout(img_bytes: bytes) -> Image.Image:
    """Extracts transparent product cutout using AI background removal (rembg) with fast studio fallback."""
    raw_img = Image.open(io.BytesIO(img_bytes))
    
    # If image already has alpha transparency (PNG), just return it
    if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
        return raw_img.convert("RGBA")
        
    try:
        import rembg
        return rembg.remove(raw_img)
    except Exception as e:
        logger.warning(f"rembg removal fallback to fast color keying: {e}")
        return fast_product_cutout(raw_img)

def compose_commercial_scene(
    scene_image: Image.Image,
    product_bytes: Optional[bytes] = None,
    logo_bytes: Optional[bytes] = None,
    brand_name: Optional[str] = None
) -> Image.Image:
    """
    Composites the authentic Sitecore product asset and official brand logo
    onto the AI-generated photorealistic scene with natural shadows and perspective.
    """
    scene = scene_image.convert("RGBA")
    sw, sh = scene.size
    
    # 1. Place Authentic Product Packshot (Hero placement on surface)
    if product_bytes:
        try:
            product = extract_product_cutout(product_bytes)
            pw, ph = product.size
            
            # Target product width: ~48% of scene width for prominent hero placement
            target_pw = int(sw * 0.48)
            aspect = ph / pw
            target_ph = int(target_pw * aspect)
            
            # If product is taller than 60% of scene, scale down to fit comfortably
            if target_ph > (sh * 0.58):
                target_ph = int(sh * 0.58)
                target_pw = int(target_ph / aspect)
                
            product_resized = product.resize((target_pw, target_ph), Image.Resampling.LANCZOS)
            
            # Position: Center-lower on table surface
            pos_x = (sw - target_pw) // 2
            pos_y = int(sh * 0.90) - target_ph
            
            # Create a soft, realistic natural contact shadow under the product base
            shadow_w = int(target_pw * 1.08)
            shadow_h = max(18, int(target_ph * 0.22))
            shadow_img = Image.new("RGBA", (shadow_w, shadow_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(shadow_img)
            # Semi-transparent dark ambient shadow
            draw.ellipse([0, 0, shadow_w, shadow_h], fill=(15, 12, 10, 130))
            shadow_blurred = shadow_img.filter(ImageFilter.GaussianBlur(radius=int(shadow_h * 0.45)))
            
            # Composite contact shadow under product base
            shadow_x = pos_x + (target_pw - shadow_w) // 2
            shadow_y = pos_y + target_ph - (shadow_h // 2)
            scene.alpha_composite(shadow_blurred, (shadow_x, shadow_y))
            
            # Composite authentic product
            scene.alpha_composite(product_resized, (pos_x, pos_y))
            logger.info(f"Composited authentic product asset ({target_pw}x{target_ph}) at ({pos_x}, {pos_y})")
        except Exception as e:
            logger.error(f"Error compositing product image: {e}")

    # 2. Place Official Brand Logo (Top-Left Brand Position)
    if logo_bytes:
        try:
            logo_raw = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
            lw, lh = logo_raw.size
            
            # Crop transparent margins from logo
            logo_bbox = logo_raw.getbbox()
            if logo_bbox:
                logo_raw = logo_raw.crop(logo_bbox)
                lw, lh = logo_raw.size
                
            # Target logo width: ~22% of scene width (e.g. ~225px for 1024 scene)
            target_lw = int(sw * 0.22)
            logo_aspect = lh / lw
            target_lh = int(target_lw * logo_aspect)
            
            # Constrain height if aspect is tall
            if target_lh > int(sh * 0.14):
                target_lh = int(sh * 0.14)
                target_lw = int(target_lh / logo_aspect)
                
            logo_resized = logo_raw.resize((target_lw, target_lh), Image.Resampling.LANCZOS)
            
            # Brand badge position
            margin_x = int(sw * 0.05)
            margin_y = int(sh * 0.05)
            
            # Render a crisp, elegant white pill badge for high contrast on any scene
            pad_x = 14
            pad_y = 10
            badge_w = target_lw + (pad_x * 2)
            badge_h = target_lh + (pad_y * 2)
            
            # Soft shadow under badge
            badge_shadow = Image.new("RGBA", (badge_w + 10, badge_h + 10), (0, 0, 0, 0))
            bs_draw = ImageDraw.Draw(badge_shadow)
            bs_draw.rounded_rectangle([4, 4, badge_w + 6, badge_h + 6], radius=12, fill=(0, 0, 0, 60))
            badge_shadow_blurred = badge_shadow.filter(ImageFilter.GaussianBlur(radius=4))
            scene.alpha_composite(badge_shadow_blurred, (margin_x - pad_x - 3, margin_y - pad_y - 1))
            
            # White rounded card badge
            badge = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
            badge_draw = ImageDraw.Draw(badge)
            badge_draw.rounded_rectangle([0, 0, badge_w, badge_h], radius=10, fill=(255, 255, 255, 240))
            
            scene.alpha_composite(badge, (margin_x - pad_x, margin_y - pad_y))
            scene.alpha_composite(logo_resized, (margin_x, margin_y))
            logger.info(f"Composited official brand logo ({target_lw}x{target_lh}) at ({margin_x}, {margin_y})")
        except Exception as e:
            logger.error(f"Error compositing brand logo: {e}")

    return scene.convert("RGB")
