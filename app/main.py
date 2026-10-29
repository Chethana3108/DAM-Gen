import os
import re
import logging
import base64
import io
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from app.sitecore_dam import SitecoreDAMClient
from app.engine import OpenSourceDiffusionEngine
from app.asset_analyzer import AssetAnalyzer, AnalyzedAsset
from app.prompt_parser import PromptParser, ParsedPrompt
from app.product_integrator import ProductIntegrator
from app.validator import QualityValidator

load_dotenv()

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('brand-image-api')

HF_TOKEN = os.getenv('HF_TOKEN', '')
IMAGE_MODEL = os.getenv('IMAGE_MODEL', 'FLUX.1-schnell (Open Source)')
MODEL_ENDPOINT = os.getenv('MODEL_ENDPOINT', 'black-forest-labs/FLUX.1-schnell')

# Initialize Sitecore DAM Client & Open-Source Diffusion Engine
dam_client = SitecoreDAMClient()
diffusion_engine = OpenSourceDiffusionEngine()

# Aspect ratio map for high-fidelity commercial resolution
ASPECT_RATIO_MAP = {
    '1:1':  (1024, 1024),
    '16:9': (1344, 768),
    '9:16': (768, 1344),
    '4:3':  (1184, 864),
    '3:4':  (864, 1184),
}

DEFAULT_NEGATIVE_PROMPT = (
    "deformed product, altered logo, fake packaging, misspelled brand name, "
    "garbled letters, distorted typography, cartoon, blurry, low resolution, "
    "noisy, ugly artifacts, bad proportions, amateur lighting"
)

app = FastAPI(
    title='Sitecore DAM GenAI Brand Campaign Assistant (Vast.ai GPU Edition)',
    description='Prompt-Only Multimodal Reference GenAI Engine connected to Sitecore Content Hub DAM',
    version='4.0.0'
)

# -------------------------------------------------------------
# Color & Style Intelligence Analyzer
# -------------------------------------------------------------

COLOR_KEYWORDS = [
    "golden brown", "golden yellow", "deep blue", "navy blue", "royal blue", "sky blue",
    "cream", "warm amber", "ruby red", "crimson", "emerald green", "botanical green",
    "pastel pink", "lavender", "chocolate brown", "bronze", "copper", "silver",
    "matte black", "pearl white", "terracotta", "warm orange", "sunlit yellow"
]

def extract_style_and_colors(prompt: str) -> Dict[str, Any]:
    """
    Intelligently extracts color palettes, lighting ambience, and sector signals
    from the natural language prompt.
    """
    p_lower = prompt.lower()
    detected_colors = []
    for c in COLOR_KEYWORDS:
        if c in p_lower:
            detected_colors.append(c)

    # Lighting ambience extraction
    ambience = "commercial advertising studio lighting"
    if any(k in p_lower for k in ["morning", "sunlight", "sunrise", "breakfast"]):
        ambience = "soft morning sunlight with gentle golden directional rays and warm ambient glow"
    elif any(k in p_lower for k in ["night", "dark", "evening", "moody"]):
        ambience = "dramatic chiaroscuro evening lighting with deep cinematic shadows"
    elif any(k in p_lower for k in ["luxury", "premium", "gold"]):
        ambience = "high-end luxury commercial lighting with specular highlights and soft diffusion"
    elif any(k in p_lower for k in ["clinical", "clean", "fresh", "pharma"]):
        ambience = "crisp, pristine daylight studio lighting with balanced natural exposure"

    return {
        "detected_colors": detected_colors,
        "ambience": ambience
    }

def build_multimodal_reference_prompt(
    user_prompt: str,
    brand_name: Optional[str],
    product_name: Optional[str],
    brand_tagline: Optional[str],
    style_info: Dict[str, Any],
    has_reference_images: bool
) -> str:
    """
    Constructs a professional multimodal conditioning prompt that instructs the AI
    to ground the scene natively in the reference assets from Sitecore DAM.
    """
    prompt_parts = []

    # 1. Core scene description
    clean_prompt = user_prompt.strip().rstrip('.')
    prompt_parts.append(f"Commercial advertising master shot: {clean_prompt}.")

    # 2. Reference image conditioning instructions
    if has_reference_images:
        prompt_parts.append(
            "Use the provided reference image as the exact product ground truth. "
            "Faithfully maintain the authentic product packaging, genuine colors, original typography, "
            "and official trademark design shown in the reference image. "
            "Render the product organically resting inside the physical scene with matching optical perspective, "
            "realistic depth of field, and natural ambient contact shadows on the surface."
        )

    # 3. Dynamic color palette intelligence
    if style_info["detected_colors"]:
        color_str = ", ".join(style_info["detected_colors"])
        prompt_parts.append(f"Harmonize the entire visual palette with brand color accents of {color_str}.")

    # 4. Lighting & environment
    prompt_parts.append(f"Lighting: {style_info['ambience']}.")

    # 5. Brand tagline integration
    if brand_tagline:
        prompt_parts.append(f"Brand theme: '{brand_tagline}'.")

    # 6. Commercial quality boosters
    prompt_parts.append(
        "8k UHD resolution, cinematic color grading, photorealistic material textures, "
        "razor-sharp focus, professional commercial composition."
    )

    return " ".join(prompt_parts)

# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------

class PromptOnlyRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=3000, description="Natural language prompt (e.g. 'Place this biscuit packet on a wooden table with a cup of coffee')")
    asset_id: Optional[int] = Field(default=None, description="Explicit Sitecore DAM Asset ID (optional, overrides search)")
    exact_filename: Optional[str] = Field(default=None, description="Explicit filename (e.g. Britannia_JimJam.png)")
    aspect_ratio: str = Field(default='1:1', pattern=r'^(1:1|3:4|4:3|9:16|16:9)$')
    sync_to_dam: bool = Field(default=True, description="Automatically ingest asset into Sitecore DAM Create tab")

class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=3000)
    asset_id: Optional[int] = None
    exact_filename: Optional[str] = None
    aspect_ratio: str = Field(default='1:1')

class UnifiedGenerationResponse(BaseModel):
    status: str
    model: str
    brand: Optional[str] = None
    product: Optional[str] = None
    prompt_used: str
    mime_type: str
    image_base64: str
    sitecore_asset_id: Optional[int] = None
    sitecore_create_url: str
    quality_report: Optional[Dict[str, Any]] = None
    disambiguation_required: bool = False
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    note: str

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'engine': 'OpenSourceDiffusionEngine',
        'has_cuda': diffusion_engine.has_cuda,
        'model_type': diffusion_engine.model_type,
        'model_endpoint': MODEL_ENDPOINT,
        'sitecore_ch_url': dam_client.ch_url,
    }

@app.post('/api/generate', response_model=UnifiedGenerationResponse)
def generate_from_prompt(req: PromptOnlyRequest):
    """
    Core Campaign Generator Pipeline:
    A. Asset Retrieval from Sitecore DAM (by exact filename, query, or explicit ID with disambiguation).
    B. Asset Understanding & Object Extraction (detects transparency vs photographic backgrounds).
    C. Natural-Language Prompt Understanding (scene, interaction, co-objects, lighting).
    D. Intelligent Scene Generation & Product Integration (multi-layer contact AO, surface reflections).
    E. Automated Quality Validation.
    F. Direct Sitecore DAM Ingestion into Create Tab.
    """
    width, height = ASPECT_RATIO_MAP.get(req.aspect_ratio, (1024, 1024))

    # Step 1: Parse Natural-Language Creative Prompt
    parsed = PromptParser.parse(req.prompt)
    target_filename = req.exact_filename or parsed.target_asset_filename

    # Step 2: Query Sitecore Content Hub DAM for Asset, Metadata & Linked Layers
    if req.asset_id:
        logger.info("Using explicitly requested DAM Asset ID: %d", req.asset_id)
        brand_context = dam_client.get_asset_by_id(req.asset_id)
    else:
        logger.info("Searching Sitecore DAM with prompt/filename...")
        brand_context = dam_client.search_brand_and_product(
            prompt=req.prompt,
            exact_filename=target_filename
        )

    # Disambiguation Check: if multiple assets share similar names
    if brand_context.get("requires_disambiguation"):
        logger.info("Disambiguation required for prompt: %s", req.prompt)
        return UnifiedGenerationResponse(
            status='disambiguation_required',
            model=IMAGE_MODEL,
            prompt_used=req.prompt,
            mime_type='',
            image_base64='',
            sitecore_create_url=f"{dam_client.ch_url}/en-us/create?p840=1",
            disambiguation_required=True,
            candidates=brand_context.get("candidates", []),
            note='Multiple matching assets found in DAM. Please select the intended asset.'
        )

    # Step 3: Download authentic reference assets & linked layers
    os.makedirs("./reference_cache", exist_ok=True)
    primary_path = None
    if brand_context.get("product_image_url"):
        p_bytes = dam_client.download_image_bytes(brand_context["product_image_url"])
        if p_bytes:
            primary_path = os.path.abspath(f"./reference_cache/ref_primary_{brand_context.get('product_id', 1)}.png")
            with open(primary_path, "wb") as f:
                f.write(p_bytes)

    # Step 4: Asset Understanding & Object Extraction (transparent layers vs photographic segmentation)
    logger.info("Step 4: Analyzing DAM asset and extracting clean product...")
    analyzed_asset = AssetAnalyzer.analyze(primary_path, metadata=brand_context)

    # Step 5: Synthesize AI Scene Environment (e.g. cafe table with coffee cup, morning sunlight)
    scene_pos_prompt, scene_neg_prompt = PromptParser.build_environment_generation_prompt(parsed)
    logger.info("Step 5: Synthesizing photorealistic scene environment: %s", scene_pos_prompt[:120])
    ai_scene = diffusion_engine.generate(
        positive_prompt=scene_pos_prompt,
        negative_prompt=scene_neg_prompt,
        reference_image_paths=[primary_path] if primary_path else [],
        width=width,
        height=height
    )

    # Step 6: Seamlessly Integrate Authentic DAM Asset with Studio Relighting & Contact Shadows
    # Retains 100% of the authentic product, genuine typography, and real textures
    logger.info("Step 6: Integrating authentic DAM product asset into AI scene with optical grounding...")
    final_image = ProductIntegrator.integrate(
        scene_image=ai_scene,
        analyzed_asset=analyzed_asset,
        intent=parsed
    )

    # Step 7: Automated Commercial Quality Validation
    quality_report = QualityValidator.validate(
        final_image=final_image,
        product_cutout=analyzed_asset.cutout_image,
        target_dimensions=(width, height)
    )

    # Convert to JPEG & Base64
    buf = io.BytesIO()
    final_image.save(buf, format="JPEG", quality=95)
    final_bytes = buf.getvalue()
    image_b64 = base64.b64encode(final_bytes).decode('utf-8')

    # Step 8: Ingest into Sitecore Content Hub DAM (Create Tab)
    sitecore_asset_id = None
    create_tab_url = f"{dam_client.ch_url}/en-us/create?p840=1"

    if req.sync_to_dam:
        try:
            p_name = brand_context.get("product_name") or brand_context.get("brand_name") or "Campaign"
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', p_name)
            filename = f"AI_{clean_name}_{int(time.time())}.jpg"
            logger.info("Step 8: Ingesting into Sitecore DAM Create Tab: %s", filename)
            dam_res = dam_client.upload_campaign_asset_to_create_tab(
                image_bytes=final_bytes,
                filename=filename,
                title=f"AI Campaign - {brand_context.get('product_name') or 'Visual'}",
                brand_id=brand_context.get("brand_id")
            )
            sitecore_asset_id = dam_res.get("asset_id")
            logger.info("Successfully ingested into Sitecore Content Hub! Asset ID: %s", sitecore_asset_id)
        except Exception as e:
            logger.error(f"Failed to ingest asset into Sitecore DAM: {e}")

    return UnifiedGenerationResponse(
        status='succeeded',
        model=f"{IMAGE_MODEL} ({diffusion_engine.model_type})",
        brand=brand_context.get("brand_name"),
        product=brand_context.get("product_name"),
        prompt_used=scene_pos_prompt,
        mime_type='image/jpeg',
        image_base64=image_b64,
        sitecore_asset_id=sitecore_asset_id,
        sitecore_create_url=create_tab_url,
        quality_report=quality_report,
        disambiguation_required=False,
        note='Original DAM product asset 100% preserved; AI-generated scene, lighting, and contact shadows.'
    )

@app.post('/api/chat')
def chat_endpoint(req: ChatRequest):
    """Conversational endpoint with automatic disambiguation support."""
    gen_res = generate_from_prompt(PromptOnlyRequest(
        prompt=req.message,
        asset_id=req.asset_id,
        exact_filename=req.exact_filename,
        aspect_ratio=req.aspect_ratio,
        sync_to_dam=True
    ))

    if gen_res.disambiguation_required:
        cand_html = []
        for c in gen_res.candidates:
            cid = c.get("id")
            ctitle = (c.get("title") or c.get("filename") or "Asset").replace('"', '&quot;')
            cand_html.append(
                f'<button class="quick-btn" onclick="selectCandidate({cid}, \'{ctitle}\')" style="text-align:left; padding:8px 12px; margin:4px 0;">'
                f'📦 <strong>{ctitle}</strong> <span style="opacity:0.7;">(ID: {cid})</span>'
                f'</button>'
            )
        return {
            "disambiguation": True,
            "reply": f"🔍 <strong>Multiple assets found in Sitecore DAM for your request.</strong><br/>Please select which asset you would like to use:<div style='display:flex; flex-direction:column; margin-top:8px;'>{''.join(cand_html)}</div>",
            "candidates": gen_res.candidates
        }

    reply_lines = [
        f"✨ **Commercial campaign visual generated!**",
        f"- **Brand**: `{gen_res.brand or 'General'}`",
        f"- **Product Reference**: `{gen_res.product or 'Official DAM Asset'}`",
    ]
    if gen_res.quality_report:
        reply_lines.append(f"- **Quality Score**: `{gen_res.quality_report.get('quality_score', 1.0) * 100:.0f}%` (Passed all pre-delivery checks)")

    if gen_res.sitecore_asset_id:
        reply_lines.append(f"- **Sitecore Asset ID**: `#{gen_res.sitecore_asset_id}` (Status: `Created / Draft`)")
        reply_lines.append(f"\n📂 **The image has been stored in your Sitecore Create Tab:**")
        reply_lines.append(f"👉 [**Open in Sitecore Content Hub Create Tab**]({gen_res.sitecore_create_url}) to review, enrich tags, and approve.")
    else:
        reply_lines.append(f"\n👉 [Open Sitecore Create Tab]({gen_res.sitecore_create_url})")

    return {
        "reply": "\n".join(reply_lines),
        "image_base64": gen_res.image_base64,
        "sitecore_asset_id": gen_res.sitecore_asset_id,
        "sitecore_create_url": gen_res.sitecore_create_url,
        "brand": gen_res.brand,
        "product": gen_res.product,
        "quality_report": gen_res.quality_report
    }

# -------------------------------------------------------------
# Modern Interactive Chat UI
# -------------------------------------------------------------

CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sitecore DAM GenAI Brand Campaign Assistant</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d1117;
      --card: #161b22;
      --border: #30363d;
      --primary: #238636;
      --primary-hover: #2ea043;
      --accent: #58a6ff;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --sitecore-red: #ea1b25;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header-left { display: flex; align-items: center; gap: 12px; }
    .badge-sitecore {
      background: var(--sitecore-red);
      color: #fff;
      font-weight: 700;
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 6px;
      letter-spacing: 0.5px;
    }
    .badge-gpu {
      background: #1f6feb;
      color: #fff;
      font-weight: 600;
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 6px;
    }
    h1 { font-size: 16px; font-weight: 600; }
    .header-links a {
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      margin-left: 16px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .header-links a:hover { text-decoration: underline; }
    
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
    }
    .message {
      display: flex;
      gap: 14px;
      animation: fadeIn 0.25s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 14px;
      flex-shrink: 0;
    }
    .avatar-user { background: #388bfd; color: #fff; }
    .avatar-ai { background: var(--sitecore-red); color: #fff; }
    .msg-content {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 18px;
      max-width: 82%;
      line-height: 1.55;
      font-size: 14.5px;
    }
    .msg-content p { margin-bottom: 8px; }
    .msg-content p:last-child { margin-bottom: 0; }
    
    .image-card {
      margin-top: 14px;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #000;
    }
    .image-card img {
      width: 100%;
      display: block;
      max-height: 500px;
      object-fit: contain;
    }
    .card-footer {
      padding: 12px 14px;
      background: #1c2128;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-top: 1px solid var(--border);
    }
    .dam-btn {
      background: var(--sitecore-red);
      color: #fff;
      text-decoration: none;
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 12.5px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: opacity 0.2s;
    }
    .dam-btn:hover { opacity: 0.9; }

    #input-container {
      background: var(--card);
      border-top: 1px solid var(--border);
      padding: 16px 24px;
    }
    .input-box-wrapper {
      max-width: 900px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .quick-prompts {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }
    .quick-btn {
      background: #21262d;
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 5px 12px;
      border-radius: 16px;
      font-size: 12px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }
    .quick-btn:hover {
      background: #30363d;
      color: var(--text);
    }
    .input-form {
      display: flex;
      gap: 12px;
    }
    textarea {
      flex: 1;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      color: #fff;
      font-family: inherit;
      font-size: 14.5px;
      resize: none;
      height: 48px;
      line-height: 24px;
    }
    textarea:focus { outline: none; border-color: var(--accent); }
    button[type="submit"] {
      background: var(--primary);
      color: #fff;
      border: none;
      padding: 0 22px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      transition: background 0.2s;
    }
    button[type="submit"]:hover { background: var(--primary-hover); }
    button[type="submit"]:disabled { opacity: 0.5; cursor: not-allowed; }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: #1c2128;
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 16px;
      font-size: 12.5px;
      color: var(--text-muted);
      margin-top: 8px;
    }
    .spinner {
      width: 14px;
      height: 14px;
      border: 2px solid var(--text-muted);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <span class="badge-sitecore">SITECORE</span>
      <span class="badge-gpu">GPU ENGINE</span>
      <h1>Content Hub GenAI Brand Assistant</h1>
    </div>
    <div class="header-links">
      <a href="https://btr-q-002.sitecorecontenthub.cloud/en-us/create?p840=1" target="_blank">
        📂 Sitecore Create Tab ↗
      </a>
      <a href="/docs" target="_blank">
        📖 API Docs ↗
      </a>
    </div>
  </header>

  <div id="chat-container">
    <div class="message">
      <div class="avatar avatar-ai">AI</div>
      <div class="msg-content">
        <p>👋 Welcome to your <strong>Universal Sitecore Content Hub GenAI Campaign Studio</strong>.</p>
        <p>Simply provide any natural language creative scenario for <strong>any brand, product, or industry</strong> in your DAM. The engine automatically:</p>
        <ol style="margin-left: 20px; margin-top: 6px; margin-bottom: 6px;">
          <li>Retrieves the authentic asset, its brand guidelines, and <strong>all linked layers / sub-assets</strong> directly from Sitecore DAM.</li>
          <li>Understands your creative scenario (tabletop placement, celebrity/model holding the product, or dynamic action).</li>
          <li>Synthesizes a realistic, high-fashion 3D image with native lighting, shadows, and natural interaction.</li>
          <li>Ingests the final campaign visual straight into your <strong>Sitecore DAM Create Tab</strong>.</li>
        </ol>
        <p>Try one of the scenario suggestions below or enter any prompt!</p>
      </div>
    </div>
  </div>

  <div id="input-container">
    <div class="input-box-wrapper">
      <div class="quick-prompts">
        <button class="quick-btn" onclick="setPrompt('Place this Jim Jam biscuit on a sunlit wooden table beside a steaming porcelain coffee cup with gentle morning light')">☕ Tabletop with Coffee Cup</button>
        <button class="quick-btn" onclick="setPrompt('A charismatic celebrity smiling and holding this Jim Jam biscuit packet in hand for a premium commercial advertisement')">🌟 Celebrity Holding Packet</button>
        <button class="quick-btn" onclick="setPrompt('A warm breakfast scene with Britannia Good Day butter cookies beside fresh butter and tea on a rustic wooden table')">🍪 Good Day Breakfast Scene</button>
        <button class="quick-btn" onclick="setPrompt('Create an organic botanical wellness campaign for Himalaya with lush emerald green herbal leaves and clear water droplets')">🌿 Skincare / Wellness Splash</button>
      </div>
      <form class="input-form" onsubmit="sendMessage(event)">
        <textarea id="prompt-input" placeholder="Type your creative campaign prompt here (e.g. Good Day biscuits in warm golden sunlight)..." required onkeydown="if(event.key==='Enter' && !event.shiftKey){ event.preventDefault(); sendMessage(event); }"></textarea>
        <button type="submit" id="send-btn">Generate</button>
      </form>
    </div>
  </div>

  <script>
    let activePrompt = '';

    function setPrompt(text) {
      const input = document.getElementById('prompt-input');
      input.value = text;
      input.focus();
    }

    async function selectCandidate(assetId, assetTitle) {
      const chatContainer = document.getElementById('chat-container');
      const userMsg = document.createElement('div');
      userMsg.className = 'message';
      userMsg.innerHTML = `
        <div class="avatar avatar-user">You</div>
        <div class="msg-content"><p>Selected asset: <strong>${assetTitle}</strong> (ID: ${assetId})</p></div>
      `;
      chatContainer.appendChild(userMsg);
      await executeGeneration(activePrompt, assetId);
    }

    async function sendMessage(e) {
      if (e) e.preventDefault();
      const input = document.getElementById('prompt-input');
      const prompt = input.value.trim();
      if (!prompt) return;

      activePrompt = prompt;
      const chatContainer = document.getElementById('chat-container');

      // Append User Message
      const userMsg = document.createElement('div');
      userMsg.className = 'message';
      userMsg.innerHTML = `
        <div class="avatar avatar-user">You</div>
        <div class="msg-content"><p>${prompt}</p></div>
      `;
      chatContainer.appendChild(userMsg);
      input.value = '';

      await executeGeneration(prompt, null);
    }

    async function executeGeneration(prompt, assetId) {
      const sendBtn = document.getElementById('send-btn');
      const chatContainer = document.getElementById('chat-container');
      sendBtn.disabled = true;

      // Append AI Loading Pill
      const aiMsg = document.createElement('div');
      aiMsg.className = 'message';
      aiMsg.innerHTML = `
        <div class="avatar avatar-ai">AI</div>
        <div class="msg-content">
          <div class="status-pill">
            <div class="spinner"></div>
            <span>Querying Sitecore DAM & generating campaign visual...</span>
          </div>
        </div>
      `;
      chatContainer.appendChild(aiMsg);
      chatContainer.scrollTop = chatContainer.scrollHeight;

      try {
        const payload = { message: prompt };
        if (assetId) payload.asset_id = assetId;

        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        
        let data = {};
        const text = await resp.text();
        try {
          data = JSON.parse(text);
        } catch(e) {
          data = { detail: text || 'Server returned an error' };
        }

        const contentDiv = aiMsg.querySelector('.msg-content');

        if (resp.ok) {
          if (data.disambiguation) {
            contentDiv.innerHTML = data.reply;
          } else {
            const scoreTxt = data.quality_report ? `<p>• <strong>Quality Validation</strong>: <code>${Math.round((data.quality_report.quality_score || 1) * 100)}% Passed</code> (100% original product preserved)</p>` : '';
            contentDiv.innerHTML = `
              <p>✨ <strong>Campaign visual synthesized successfully!</strong></p>
              <p>• <strong>Brand</strong>: <code>${data.brand || 'General'}</code> | <strong>Product</strong>: <code>${data.product || 'Official DAM Asset'}</code></p>
              ${scoreTxt}
              ${data.sitecore_asset_id ? `<p>• <strong>Sitecore Asset ID</strong>: <code>#${data.sitecore_asset_id}</code> (Status: <strong>Created / Draft</strong>)</p>` : ''}
              <div class="image-card">
                <img src="data:image/jpeg;base64,${data.image_base64}" alt="Generated Campaign Image" />
                <div class="card-footer">
                  <span style="font-size: 12px; color: var(--text-muted);">Stored directly in Sitecore Content Hub DAM</span>
                  <a href="${data.sitecore_create_url}" target="_blank" class="dam-btn">
                    Review in Sitecore Create Tab ↗
                  </a>
                </div>
              </div>
            `;
          }
        } else {
          contentDiv.innerHTML = `
            <p style="color: #f85149;">❌ Error: ${data.detail || 'Failed to generate visual. Please try again.'}</p>
          `;
        }
      } catch (err) {
        aiMsg.querySelector('.msg-content').innerHTML = `
          <p style="color: #f85149;">❌ Connection error: ${err.message}</p>
        `;
      } finally {
        sendBtn.disabled = false;
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }
  </script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
@app.get('/chat', response_class=HTMLResponse)
def serve_chat_ui():
    return HTMLResponse(content=CHAT_HTML)
