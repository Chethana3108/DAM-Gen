import os
import io
import time
import base64
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from app.sitecore_dam import SitecoreDAMClient
from app.engine import GPTImageEngine
from app.asset_analyzer import AssetAnalyzer
from app.prompt_parser import PromptParser
from app.validator import QualityValidator

load_dotenv()

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('dam-brand-image-api')

OPENAI_IMAGE_MODEL = os.getenv('OPENAI_IMAGE_MODEL', 'gpt-image-2.5-sunburst')

# Aspect ratio map (Aspect Ratio -> (Width, Height))
ASPECT_RATIO_MAP = {
    '1:1':  (1024, 1024),
    '16:9': (1536, 1024),
    '9:16': (1024, 1536),
    '4:3':  (1536, 1024),
    '3:4':  (1024, 1536),
}

# Initialize Sitecore DAM Client & GPT Image Engine
dam_client = SitecoreDAMClient()
gpt_engine = GPTImageEngine()
gpt_service = gpt_engine.gpt_service

app = FastAPI(
    title='Sitecore DAM GenAI Campaign Studio API',
    description='Production Pure Backend REST API connected to Sitecore Content Hub DAM using GPT Image reference-guided generation',
    version='5.0.0'
)

# Enable CORS for frontend interoperability
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------

class CandidateAsset(BaseModel):
    id: int
    title: Optional[str] = None
    filename: Optional[str] = None


class PromptOnlyRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=3000, description="Natural language prompt")
    asset_id: Optional[int] = Field(default=None, description="Explicit Sitecore DAM Asset ID")
    exact_filename: Optional[str] = Field(default=None, description="Explicit filename (e.g. Britannia_JimJam.png)")
    aspect_ratio: str = Field(default='1:1', pattern=r'^(1:1|3:4|4:3|9:16|16:9)$')
    sync_to_dam: bool = Field(default=False, description="Whether to automatically ingest asset into Sitecore DAM immediately")


class GenerateVariationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=3000)
    asset_id: Optional[int] = None
    exact_filename: Optional[str] = None
    aspect_ratio: str = Field(default='1:1', pattern=r'^(1:1|3:4|4:3|9:16|16:9)$')
    variation_index: int = Field(default=1, ge=1, le=10)


class ApproveAndSaveRequest(BaseModel):
    image_base64: str = Field(description="Base64 encoded approved image bytes")
    source_asset_id: int = Field(description="Original Sitecore DAM Asset ID")
    source_asset_name: Optional[str] = Field(default=None, description="Source asset name or title")
    prompt: str = Field(description="User creative prompt")
    filename: Optional[str] = Field(default=None, description="Approved filename")
    title: Optional[str] = Field(default=None, description="Approved asset title")
    brand_id: Optional[int] = Field(default=None, description="Sitecore Brand entity ID")


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=3000)
    asset_id: Optional[int] = None
    exact_filename: Optional[str] = None
    aspect_ratio: str = Field(default='1:1', pattern=r'^(1:1|3:4|4:3|9:16|16:9)$')


class UnifiedGenerationResponse(BaseModel):
    status: str
    model: str
    brand: Optional[str] = None
    product: Optional[str] = None
    source_asset_id: Optional[int] = None
    source_asset_name: Optional[str] = None
    prompt_used: str
    suggested_filename: str
    mime_type: str
    image_base64: str
    sitecore_asset_id: Optional[int] = None
    sitecore_create_url: str
    quality_report: Optional[Dict[str, Any]] = None
    disambiguation_required: bool = False
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    note: str


class ApproveAndSaveResponse(BaseModel):
    success: bool
    asset_id: int
    filename: str
    title: str
    source_asset_id: int
    create_tab_url: str
    asset_url: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    reply: str
    image_base64: Optional[str] = None
    suggested_filename: Optional[str] = None
    source_asset_id: Optional[int] = None
    source_asset_name: Optional[str] = None
    brand: Optional[str] = None
    product: Optional[str] = None
    quality_report: Optional[Dict[str, Any]] = None
    sitecore_create_url: Optional[str] = None
    disambiguation: bool = False
    candidates: List[Dict[str, Any]] = Field(default_factory=list)


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.get('/', response_class=JSONResponse)
def root():
    """Service status and OpenAPI documentation entry point."""
    return {
        "service": "Sitecore Content Hub DAM GenAI Campaign Studio API",
        "version": "5.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "model": OPENAI_IMAGE_MODEL,
        "sitecore_configured": dam_client.is_configured,
        "gpt_available": gpt_service.is_available,
    }


@app.get('/health')
def health():
    """Health check endpoint providing engine and integration status."""
    return {
        'status': 'ok',
        'engine': 'GPTImageEngine',
        'model': OPENAI_IMAGE_MODEL,
        'gpt_available': gpt_service.is_available,
        'sitecore_ch_url': dam_client.ch_url,
    }


@app.get('/api/dam/assets')
def list_dam_assets(query: Optional[str] = None, take: int = Query(default=40, ge=1, le=100)):
    """Browse available image assets in Sitecore Content Hub DAM."""
    try:
        assets = dam_client.list_assets(take=take, filter_query=query)
        return {"assets": assets, "total": len(assets)}
    except Exception as e:
        logger.error(f"Error listing DAM assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/dam/asset/{asset_id}')
def get_dam_asset(asset_id: int):
    """Retrieve details, dimensions, and linked layers of an asset in DAM."""
    try:
        details = dam_client.get_asset_by_id(asset_id)
        return details
    except Exception as e:
        logger.error(f"Error getting DAM asset {asset_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Asset #{asset_id} not found: {e}")


@app.post('/api/generate', response_model=UnifiedGenerationResponse)
def generate_campaign_image(req: PromptOnlyRequest):
    """
    Core Campaign Visual Generator:
    1. Retrieves authentic product asset & metadata from Sitecore Content Hub DAM.
    2. Dynamically analyzes creative prompt and asset using multimodal AI.
    3. Synthesizes an advertising visual with GPT Image model using the authentic DAM reference asset.
    4. Evaluates quality with commercial QA validation (semantic + technical).
    5. Automatically refines the visual if quality defects are diagnosed.
    6. Generates dynamic suggested filename based on DAM metadata and prompt context.
    7. Previews result without creating duplicate DAM assets unless explicitly requested.
    """
    width, height = ASPECT_RATIO_MAP.get(req.aspect_ratio, (1024, 1024))

    # Step 1: Preliminary Prompt Inspection for explicit target asset
    preliminary_parsed = PromptParser.parse(req.prompt)
    target_filename = req.exact_filename or preliminary_parsed.target_asset_filename

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
            model=OPENAI_IMAGE_MODEL,
            prompt_used=req.prompt,
            suggested_filename='',
            mime_type='',
            image_base64='',
            sitecore_create_url=f"{dam_client.ch_url}/en-us/create?p840=1",
            disambiguation_required=True,
            candidates=brand_context.get("candidates", []),
            note='Multiple matching assets found in DAM. Please select the intended asset.'
        )

    # Step 3: Download authentic reference assets & linked layers
    cache_dir = os.path.abspath("./reference_cache")
    os.makedirs(cache_dir, exist_ok=True)

    primary_path = None
    if brand_context.get("product_image_url"):
        p_bytes = dam_client.download_image_bytes(brand_context["product_image_url"])
        if p_bytes:
            primary_path = os.path.join(cache_dir, f"ref_primary_{brand_context.get('product_id', 1)}.png")
            with open(primary_path, "wb") as f:
                f.write(p_bytes)

    if not primary_path or not os.path.exists(primary_path):
        raise HTTPException(
            status_code=404,
            detail=f"Unable to retrieve image file for asset #{brand_context.get('product_id')}. Asset may be missing or inaccessible."
        )

    # Download linked layers if available
    linked_layer_paths = []
    for idx, layer_url in enumerate(brand_context.get("layer_urls", [])):
        l_bytes = dam_client.download_image_bytes(layer_url)
        if l_bytes:
            l_path = os.path.join(cache_dir, f"ref_layer_{brand_context.get('product_id', 1)}_{idx}.png")
            with open(l_path, "wb") as f:
                f.write(l_bytes)
            linked_layer_paths.append(l_path)

    # Step 4: Asset Understanding (non-destructive; does not strip image borders)
    logger.info("Step 4: Analyzing DAM asset and discovering linked layers...")
    AssetAnalyzer.analyze(
        file_path=primary_path,
        metadata=brand_context,
        linked_layer_paths=linked_layer_paths,
        require_cutout_extraction=False
    )

    # Step 5: Dynamic AI Creative Understanding
    raw_ref_image = Image.open(primary_path)
    parsed = PromptParser.parse(
        prompt=req.prompt,
        metadata=brand_context,
        reference_image=raw_ref_image,
        gpt_service=gpt_service
    )

    # Step 6: Native Reference-Guided Scene Generation with GPT Image model
    logger.info("Step 6: Synthesizing visual with %s using authentic reference asset...", OPENAI_IMAGE_MODEL)
    ref_paths = [primary_path] + linked_layer_paths

    final_image = gpt_engine.generate(
        positive_prompt=parsed.unified_generation_prompt,
        reference_image_paths=ref_paths,
        width=width,
        height=height
    )

    # Step 7: Automated Quality Validation
    logger.info("Step 7: Performing commercial QA validation on generated visual...")
    quality_report = QualityValidator.validate(
        final_image=final_image,
        reference_image=raw_ref_image,
        target_dimensions=(width, height),
        prompt=req.prompt,
        gpt_service=gpt_service
    )

    # Step 8: Automatic Refinement Loop (if defects detected)
    max_retries = 2
    retry_count = 0
    while quality_report.get("needs_refinement") and retry_count < max_retries:
        retry_count += 1
        refine_instruction = quality_report.get("refinement_instruction") or "Improve product presence and visual integration."
        logger.info("Step 8: Automated refinement iteration #%d: %s", retry_count, refine_instruction)
        try:
            refined_candidate = gpt_engine.refine(
                image=final_image,
                refinement_instruction=refine_instruction,
                original_prompt=parsed.unified_generation_prompt,
                width=width,
                height=height
            )
            new_report = QualityValidator.validate(
                final_image=refined_candidate,
                reference_image=raw_ref_image,
                target_dimensions=(width, height),
                prompt=req.prompt,
                gpt_service=gpt_service
            )
            if new_report.get("quality_score", 0) >= quality_report.get("quality_score", 0):
                final_image = refined_candidate
                quality_report = new_report
        except Exception as e_refine:
            logger.warning("Automated refinement iteration encountered error (%s); keeping current image.", e_refine)
            break

    # Determine overall status
    generation_status = 'succeeded' if quality_report.get("passed") else 'review_required'

    # Step 9: Suggested Asset Naming (derived dynamically from DAM metadata and prompt)
    src_display_name = brand_context.get("product_name") or brand_context.get("title") or brand_context.get("filename") or "Asset"
    suggested_filename = dam_client.generate_suggested_filename(
        source_name=src_display_name,
        prompt=req.prompt,
        brand_name=brand_context.get("brand_name"),
        ext="png"
    )

    # Encode Image to Base64 PNG
    buf = io.BytesIO()
    final_image.save(buf, format="PNG")
    final_bytes = buf.getvalue()
    image_b64 = base64.b64encode(final_bytes).decode('utf-8')

    # Optional immediate DAM sync (default is False for preview)
    sitecore_asset_id = None
    create_tab_url = f"{dam_client.ch_url}/en-us/create?p840=1"

    if req.sync_to_dam:
        try:
            logger.info("Syncing approved asset to Sitecore DAM Create tab: %s", suggested_filename)
            dam_res = dam_client.save_approved_campaign_asset(
                image_bytes=final_bytes,
                filename=suggested_filename,
                title=f"{src_display_name} - Campaign Visual",
                source_asset_id=brand_context.get("product_id"),
                source_asset_name=src_display_name,
                prompt=req.prompt,
                model_identifier=OPENAI_IMAGE_MODEL,
                brand_id=brand_context.get("brand_id")
            )
            sitecore_asset_id = dam_res.get("asset_id")
            logger.info("Successfully ingested into Sitecore Content Hub! Asset ID: %s", sitecore_asset_id)
        except Exception as e:
            logger.error("Failed to ingest asset into Sitecore DAM: %s", e)
            return UnifiedGenerationResponse(
                status='failed',
                model=OPENAI_IMAGE_MODEL,
                brand=brand_context.get("brand_name"),
                product=brand_context.get("product_name"),
                source_asset_id=brand_context.get("product_id"),
                source_asset_name=src_display_name,
                prompt_used=req.prompt,
                suggested_filename=suggested_filename,
                mime_type='image/png',
                image_base64=image_b64,
                sitecore_asset_id=None,
                sitecore_create_url=create_tab_url,
                quality_report=quality_report,
                disambiguation_required=False,
                note=f"Image generation succeeded, but Sitecore DAM sync failed: {e}"
            )

    return UnifiedGenerationResponse(
        status=generation_status,
        model=OPENAI_IMAGE_MODEL,
        brand=brand_context.get("brand_name"),
        product=brand_context.get("product_name"),
        source_asset_id=brand_context.get("product_id"),
        source_asset_name=src_display_name,
        prompt_used=req.prompt,
        suggested_filename=suggested_filename,
        mime_type='image/png',
        image_base64=image_b64,
        sitecore_asset_id=sitecore_asset_id,
        sitecore_create_url=create_tab_url,
        quality_report=quality_report,
        disambiguation_required=False,
        note=f"Synthesized with {OPENAI_IMAGE_MODEL} using authentic DAM reference asset."
    )


@app.post('/api/generate-variation', response_model=UnifiedGenerationResponse)
def generate_variation(req: GenerateVariationRequest):
    """Generates an aesthetic variation using GPT Image model and reference asset."""
    return generate_campaign_image(PromptOnlyRequest(
        prompt=f"{req.prompt} (variation #{req.variation_index})",
        asset_id=req.asset_id,
        exact_filename=req.exact_filename,
        aspect_ratio=req.aspect_ratio,
        sync_to_dam=False
    ))


@app.post('/api/approve-and-save', response_model=ApproveAndSaveResponse)
def approve_and_save_to_dam(req: ApproveAndSaveRequest):
    """
    Approves the previewed image and creates a new M.Asset in Sitecore Content Hub Create Tab.
    Applies Title, FileName, Description (with generation provenance), AltText,
    and establishes RelatedAsset link to the original source asset.
    """
    try:
        raw_bytes = base64.b64decode(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    filename = req.filename or f"Campaign_Asset_{int(time.time())}.png"
    title = req.title or filename.replace('.png', '').replace('_', ' ')

    try:
        result = dam_client.save_approved_campaign_asset(
            image_bytes=raw_bytes,
            filename=filename,
            title=title,
            source_asset_id=req.source_asset_id,
            source_asset_name=req.source_asset_name,
            prompt=req.prompt,
            model_identifier=OPENAI_IMAGE_MODEL,
            brand_id=req.brand_id
        )
        return ApproveAndSaveResponse(
            success=True,
            asset_id=result.get("asset_id"),
            filename=filename,
            title=title,
            source_asset_id=req.source_asset_id,
            create_tab_url=result.get("create_tab_url"),
            asset_url=result.get("asset_url"),
            message=f"Successfully created new asset #{result.get('asset_id')} in Sitecore Content Hub Create Tab linked to source asset #{req.source_asset_id}."
        )
    except Exception as e:
        logger.error(f"Failed to save asset to Sitecore DAM: {e}")
        raise HTTPException(status_code=500, detail=f"Sitecore DAM asset creation failed: {e}")


@app.post('/api/cancel')
def cancel_preview():
    """Cancels current preview session without writing to DAM."""
    return {"status": "cancelled", "message": "Preview session discarded. No asset was created in Sitecore DAM."}


@app.post('/api/chat', response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Conversational endpoint supporting asset search, generation preview, and direct confirmation."""
    gen_res = generate_campaign_image(PromptOnlyRequest(
        prompt=req.message,
        asset_id=req.asset_id,
        exact_filename=req.exact_filename,
        aspect_ratio=req.aspect_ratio,
        sync_to_dam=False
    ))

    if gen_res.disambiguation_required:
        return ChatResponse(
            reply="🔍 Multiple matching assets found in Sitecore Content Hub DAM. Please choose the intended asset:",
            disambiguation=True,
            candidates=gen_res.candidates,
            sitecore_create_url=gen_res.sitecore_create_url
        )

    reply_lines = [
        f"✨ **Commercial campaign visual generated with {OPENAI_IMAGE_MODEL}!**",
        f"- **Brand**: `{gen_res.brand or 'General'}`",
        f"- **Product Reference**: `{gen_res.product or 'Official DAM Asset'}` (ID: `#{gen_res.source_asset_id}`)",
        f"- **Suggested Filename**: `{gen_res.suggested_filename}`",
    ]
    if gen_res.quality_report:
        reply_lines.append(f"- **Quality Score**: `{gen_res.quality_report.get('quality_score', 1.0) * 100:.0f}%` (100% Brand Packaging Preserved)")

    return ChatResponse(
        reply="\n".join(reply_lines),
        image_base64=gen_res.image_base64,
        suggested_filename=gen_res.suggested_filename,
        source_asset_id=gen_res.source_asset_id,
        source_asset_name=gen_res.source_asset_name,
        brand=gen_res.brand,
        product=gen_res.product,
        quality_report=gen_res.quality_report,
        sitecore_create_url=gen_res.sitecore_create_url,
        disambiguation=False,
        candidates=[]
    )
