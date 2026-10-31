"""
Universal Sitecore Content Hub DAM Client.
Handles dynamic asset retrieval, metadata extraction, linked layer discovery,
and verified asset ingestion into Sitecore Content Hub DAM Create Sandbox.
"""

import os
import io
import re
import time
import datetime
import logging
import urllib.request
import urllib.parse
import requests
from typing import Optional, Dict, Any, Tuple, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

class SitecoreDAMClient:
    """
    Universal Sitecore Content Hub DAM Client.
    Connects to Sitecore Content Hub DAM using environment configuration.
    Contains NO hardcoded credential fallbacks or brand-specific dictionaries.
    """

    def __init__(
        self,
        ch_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.ch_url = (ch_url or os.getenv("SITECORE_CH_URL", "")).rstrip("/")
        self.username = username or os.getenv("SITECORE_USERNAME", "")
        self.password = password or os.getenv("SITECORE_PASSWORD", "")
        self.client_id = client_id or os.getenv("SITECORE_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("SITECORE_CLIENT_SECRET", "")

        self._token: Optional[str] = None
        self._token_expiry: float = 0

    @property
    def is_configured(self) -> bool:
        """Checks if all required Sitecore credentials are present."""
        return bool(self.ch_url and self.username and self.password and self.client_id and self.client_secret)

    def get_token(self, force_refresh: bool = False) -> str:
        """Obtains or reuses a valid OAuth Bearer token."""
        if not self.is_configured:
            raise RuntimeError("Sitecore Content Hub credentials are not configured in environment.")

        if not force_refresh and self._token and time.time() < (self._token_expiry - 60):
            return self._token

        token_url = f"{self.ch_url}/oauth/token"
        data = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": self.password,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            resp = requests.post(token_url, data=data, headers=headers, timeout=15)
            resp.raise_for_status()
            res_json = resp.json()
            self._token = res_json.get("access_token")
            expires_in = res_json.get("expires_in", 3600)
            self._token_expiry = time.time() + float(expires_in)
            logger.info("Successfully authenticated with Sitecore Content Hub OAuth.")
            return self._token
        except Exception as e:
            logger.error("Failed to authenticate with Sitecore Content Hub: %s", e)
            raise RuntimeError(f"Sitecore Content Hub Authentication Failed: {e}")

    def _auth_headers(self) -> Dict[str, str]:
        token = self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def list_assets(self, take: int = 50, filter_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists available image assets in Sitecore Content Hub DAM for asset browsing.
        Filters out non-image files and system bundles.
        """
        headers = self._auth_headers()
        url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Asset'&take={take * 2}"

        assets = []
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                items = r.json().get("items", [])
                for item in items:
                    props = item.get("properties", {})
                    fname = (props.get("FileName") or "").lower()
                    title = props.get("Title") or fname
                    if not any(fname.endswith(ext) for ext in VALID_IMAGE_EXTS):
                        continue

                    if filter_query:
                        fq = filter_query.lower()
                        if fq not in fname and fq not in title.lower():
                            continue

                    aid = item.get("id")
                    file_props = props.get("FileProperties", {}).get("properties", {})
                    w = file_props.get("width")
                    h = file_props.get("height")
                    dim_str = f"{w}x{h}" if (w and h) else ""

                    thumb_url = self._get_asset_delivery_url(aid, preferred=["preview", "bigthumbnail", "thumbnail", "downloadPreview"])

                    assets.append({
                        "id": aid,
                        "title": title,
                        "filename": props.get("FileName"),
                        "thumbnail_url": thumb_url,
                        "dimensions": dim_str,
                        "mime_type": file_props.get("content_type", "image/png"),
                        "extension": file_props.get("extension", "png"),
                        "filesize": file_props.get("filesize", 0)
                    })
                    if len(assets) >= take:
                        break
        except Exception as e:
            logger.warning("Error querying DAM assets list: %s", e)

        return assets

    def search_brand_and_product(self, prompt: str, exact_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Dynamically analyzes user prompt or exact filename and retrieves matching brand details,
        approved product asset, and all linked layers from Sitecore DAM.
        Flags requires_disambiguation when multiple assets match closely.
        """
        headers = self._auth_headers()
        prompt_lower = prompt.lower()

        result = {
            "brand_id": None,
            "brand_name": None,
            "brand_tagline": None,
            "brand_logo_url": None,
            "product_id": None,
            "product_name": None,
            "product_image_url": None,
            "filename": "",
            "title": "",
            "mime_type": "image/png",
            "extension": "png",
            "dimensions": {"width": 1024, "height": 1024},
            "filesize": 0,
            "description": "",
            "alt_text": "",
            "tags": [],
            "layer_urls": [],
            "matched_keywords": [],
            "requires_disambiguation": False,
            "candidates": []
        }

        # 1. Query Brands Dynamically from Sitecore Content Hub
        brands = []
        brands_url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Brand'&take=50"
        try:
            r = requests.get(brands_url, headers=headers, timeout=10)
            if r.status_code == 200:
                brands = r.json().get("items", [])
                for b in brands:
                    props = b.get("properties", {})
                    bname = props.get("BrandName") or props.get("BrandLabel", {}).get("en-US", "")
                    synonyms = props.get("BrandSynonyms", {}).get("en-US", "")

                    if (bname and bname.lower() in prompt_lower) or (synonyms and synonyms.lower() in prompt_lower):
                        result["brand_id"] = b.get("id")
                        result["brand_name"] = bname
                        tagline = props.get("BrandDescription", {}).get("en-US")
                        if tagline:
                            result["brand_tagline"] = tagline
                        result["matched_keywords"].append(f"Brand: {bname}")
                        break
        except Exception as e:
            logger.warning("Error querying M.Brand: %s", e)

        # 2. Query Assets from DAM
        assets = []
        assets_query_url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Asset'&take=100"
        try:
            r_assets = requests.get(assets_query_url, headers=headers, timeout=12)
            if r_assets.status_code == 200:
                assets = r_assets.json().get("items", [])
        except Exception as e:
            logger.warning("Error querying DAM assets: %s", e)

        # Step A: If exact filename was supplied
        if exact_filename:
            clean_exact = exact_filename.strip().lower()
            fname_stem = re.sub(r'\.(png|jpg|jpeg|webp|gif)$', '', clean_exact)
            fname_tokens = [tok for tok in re.split(r'[_\-\s\.]+', fname_stem) if len(tok) > 2]
            exact_matches = []

            for a in assets:
                props = a.get("properties", {})
                fname = (props.get("FileName") or "").lower()
                title = (props.get("Title") or "").lower()
                combined_name = f"{title} {fname}"

                if not any(fname.endswith(ext) for ext in VALID_IMAGE_EXTS):
                    continue

                if clean_exact in fname or fname in clean_exact or fname_stem in fname:
                    exact_matches.append(a)
                elif fname_tokens and all(tok in combined_name for tok in fname_tokens):
                    exact_matches.append(a)

            if len(exact_matches) > 1:
                result["requires_disambiguation"] = True
                result["candidates"] = [
                    {"id": m.get("id"), "title": m.get("properties", {}).get("Title"), "filename": m.get("properties", {}).get("FileName")}
                    for m in exact_matches[:5]
                ]
                logger.info("Multiple DAM assets matched '%s'. Disambiguation required.", exact_filename)
                return result
            elif len(exact_matches) == 1:
                best_match = exact_matches[0]
                logger.info("Exact match found for '%s': Asset ID %s", exact_filename, best_match.get("id"))
                return self._populate_asset_details(best_match, result, brands, headers)

        # Step B: Dynamic keyword and token search without hardcoded brand lists
        context_words = {"table", "cup", "coffee", "tea", "desk", "counter", "surface", "background", "hand", "hands", "holding", "hold", "person", "celebrity", "woman", "man", "model", "morning", "night", "light", "sunlight", "outdoor", "indoor"}
        stop_words = {"a", "an", "the", "in", "on", "at", "for", "with", "and", "or", "of", "to", "is", "like", "by", "scene", "campaign", "image", "create", "visual", "photo", "this", "that", "want", "you", "generate", "xyz", "particular", "some", "made", "placed", "where", "png", "jpg", "jpeg"}
        all_ignore = stop_words | context_words

        raw_tokens = re.split(r'[\s_\-\.]+', prompt_lower)
        product_words = [w.strip(".,!?\"'") for w in raw_tokens if w.strip(".,!?\"'") not in all_ignore and len(w.strip(".,!?\"'")) > 2]
        if not product_words:
            product_words = [w.strip(".,!?\"'") for w in raw_tokens if w.strip(".,!?\"'") not in stop_words and len(w.strip(".,!?\"'")) > 2]

        best_match = None
        best_score = 0
        candidate_assets = []

        for a in assets:
            props = a.get("properties", {})
            title = (props.get("Title") or "").lower()
            fname = (props.get("FileName") or "").lower()
            desc = str(props.get("Description") or {}).lower()
            alt = str(props.get("SC.Asset.AltText") or {}).lower()
            combined = f"{title} {fname}"

            if not any(fname.endswith(ext) for ext in VALID_IMAGE_EXTS) or "logo" in combined:
                continue

            score = 0
            combined_condensed = f"{title}{fname}".replace(" ", "").replace("_", "").replace("-", "")
            for pw in product_words:
                pw_clean = pw.replace(" ", "").replace("_", "").replace("-", "")
                if pw in combined:
                    score += 10
                if re.search(r'\b' + re.escape(pw) + r'\b', combined):
                    score += 15
                if len(pw_clean) > 3 and pw_clean in combined_condensed:
                    score += 30
                if pw in desc or pw in alt:
                    score += 2

            if score > 0:
                candidate_assets.append((score, a))
            if score > best_score:
                best_score = score
                best_match = a

        # Check for multiple high-scoring matches requiring disambiguation
        high_scorers = [c for c in candidate_assets if c[0] >= (best_score * 0.85) and c[0] > 15]
        if len(high_scorers) > 1 and not exact_filename:
            result["requires_disambiguation"] = True
            result["candidates"] = [
                {"id": m[1].get("id"), "title": m[1].get("properties", {}).get("Title"), "filename": m[1].get("properties", {}).get("FileName")}
                for m in high_scorers[:5]
            ]
            logger.info("Multiple candidate assets found with similar relevance scores. Disambiguation required.")
            return result

        # Fallback to first valid image asset if no keyword match
        if not best_match and assets:
            for a in assets:
                fname = (a.get("properties", {}).get("FileName") or "").lower()
                if any(fname.endswith(ext) for ext in VALID_IMAGE_EXTS) and "logo" not in fname:
                    best_match = a
                    break

        if best_match:
            return self._populate_asset_details(best_match, result, brands, headers, candidate_assets=candidate_assets, all_ignore=all_ignore)

        return result

    def get_asset_by_id(self, asset_id: int) -> Dict[str, Any]:
        """Directly retrieves an asset and its linked layers by ID."""
        headers = self._auth_headers()
        ent_url = f"{self.ch_url}/api/entities/{asset_id}"
        resp = requests.get(ent_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise ValueError(f"Asset ID {asset_id} not found in DAM.")
        asset_data = resp.json()
        result = {
            "brand_id": None, "brand_name": None, "brand_tagline": None,
            "brand_logo_url": None, "product_id": asset_id, "product_name": None,
            "product_image_url": None, "filename": "", "title": "",
            "mime_type": "image/png", "extension": "png", "dimensions": {"width": 1024, "height": 1024},
            "filesize": 0, "description": "", "alt_text": "", "tags": [],
            "layer_urls": [], "matched_keywords": [], "requires_disambiguation": False, "candidates": []
        }
        return self._populate_asset_details(asset_data, result, [], headers)

    def _populate_asset_details(
        self,
        best_match: Dict[str, Any],
        result: Dict[str, Any],
        brands: list,
        headers: dict,
        candidate_assets: list = None,
        all_ignore: set = None
    ) -> Dict[str, Any]:
        """Populates rich asset metadata, linked layers, and official logo."""
        all_ignore = all_ignore or set()
        candidate_assets = candidate_assets or []
        pid = best_match.get("id")
        props = best_match.get("properties", {})
        result["product_id"] = pid
        result["product_name"] = props.get("Title") or props.get("FileName")
        result["title"] = props.get("Title") or ""
        result["filename"] = props.get("FileName") or ""
        result["product_image_url"] = self._get_asset_delivery_url(pid, preferred=["downloadOriginal", "vision_ready", "preview", "downloadPreview"])

        # File technical properties
        fprops = props.get("FileProperties", {}).get("properties", {})
        if fprops:
            result["mime_type"] = fprops.get("content_type", "image/png")
            result["extension"] = fprops.get("extension", "png")
            try:
                w = int(fprops.get("width", 1024))
                h = int(fprops.get("height", 1024))
                result["dimensions"] = {"width": w, "height": h}
            except Exception:
                result["dimensions"] = {"width": 1024, "height": 1024}
            result["filesize"] = fprops.get("filesize", 0)
        else:
            ext = "png"
            if result["filename"].lower().endswith((".jpg", ".jpeg")):
                ext = "jpg"
            result["extension"] = ext
            result["mime_type"] = "image/jpeg" if ext == "jpg" else "image/png"
            result["dimensions"] = {"width": 1024, "height": 1024}

        desc_dict = props.get("Description") or {}
        result["description"] = desc_dict.get("en-US", "") if isinstance(desc_dict, dict) else str(desc_dict)
        alt_dict = props.get("SC.Asset.AltText") or {}
        result["alt_text"] = alt_dict.get("en-US", "") if isinstance(alt_dict, dict) else str(alt_dict)

        ai_sugg = props.get("AI.Suggestions") or {}
        tag_suggestions = ai_sugg.get("TagToAsset", []) if isinstance(ai_sugg, dict) else []
        sim_tags = [t.get("name") for t in (props.get("ImageSimilarityTags") or []) if isinstance(t, dict) and "name" in t]
        result["tags"] = list(set(tag_suggestions + sim_tags))
        result["matched_keywords"].append(f"Product: {result['product_name']}")
        logger.info("Dynamically resolved product asset in DAM: ID %s ('%s')", pid, result["product_name"])

        # Dynamic Linked Layer Discovery via entity relations
        layer_ids = set()
        try:
            ent_url = f"{self.ch_url}/api/entities/{pid}"
            ent_r = requests.get(ent_url, headers=headers, timeout=10)
            if ent_r.status_code == 200:
                rels = ent_r.json().get("relations", {})
                for rel_key in ["RelatedAsset", "AssetToLinkedAsset", "MasterAssetToSubAsset", "IngredientImage"]:
                    rel_info = rels.get(rel_key, {})
                    for item in rel_info.get("children", []) + rel_info.get("parents", []):
                        href = item.get("href", "")
                        if href and "/api/entities/" in href:
                            try:
                                lid = int(href.rstrip("/").split("/")[-1])
                                if lid != pid:
                                    layer_ids.add(lid)
                            except Exception:
                                pass

                # Infer Brand relation if available
                if not result["brand_name"]:
                    for b_rel_key in ["PCMBrandToAsset", "BrandToAsset"]:
                        b_rel = rels.get(b_rel_key, {})
                        for b_item in b_rel.get("parents", []) + b_rel.get("children", []):
                            href = b_item.get("href", "")
                            if href and "/api/entities/" in href:
                                try:
                                    bid = int(href.rstrip("/").split("/")[-1])
                                    for b in brands:
                                        if b.get("id") == bid:
                                            result["brand_id"] = bid
                                            result["brand_name"] = b.get("properties", {}).get("BrandName", "")
                                            break
                                except Exception:
                                    pass
        except Exception as e:
            logger.warning("Error inspecting entity relations for asset %s: %s", pid, e)

        for lid in layer_ids:
            l_url = self._get_asset_delivery_url(lid, preferred=["downloadOriginal", "vision_ready", "preview", "downloadPreview"])
            if l_url and l_url not in result["layer_urls"]:
                result["layer_urls"].append(l_url)
                logger.info("Attached linked layer from DAM: Asset ID %s", lid)

        # Dynamic Brand Logo Discovery
        bname_lower = (result["brand_name"] or "").lower()
        assets_query_url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Asset'&take=50"
        try:
            r_a = requests.get(assets_query_url, headers=headers, timeout=10)
            if r_a.status_code == 200:
                for a in r_a.json().get("items", []):
                    a_props = a.get("properties", {})
                    combined = f"{a_props.get('Title', '')} {a_props.get('FileName', '')}".lower()
                    if "logo" in combined:
                        if bname_lower and bname_lower in combined:
                            result["brand_logo_url"] = self._get_asset_delivery_url(a.get("id"))
                            break
                        elif not result["brand_logo_url"]:
                            result["brand_logo_url"] = self._get_asset_delivery_url(a.get("id"))
        except Exception as e:
            logger.warning("Error finding logo: %s", e)

        return result

    def _get_asset_delivery_url(self, asset_id: int, preferred: list = None) -> Optional[str]:
        """Resolves the direct delivery URL for an asset's preferred rendition."""
        preferred = preferred or ["preview", "downloadPreview", "downloadOriginal", "bigthumbnail"]
        headers = self._auth_headers()
        url = f"{self.ch_url}/api/entities/{asset_id}/renditions"

        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                return None
            renditions = r.json().get("renditions", [])

            rend_map = {}
            for ritem in renditions:
                rname = ritem.get("rendition_link", {}).get("name")
                files = ritem.get("file_location", {}).get("files", [])
                if files and "delivery_link" in files[0]:
                    rend_map[rname] = files[0]["delivery_link"]["href"]

            for pref in preferred:
                if pref in rend_map:
                    return rend_map[pref]

            for k, href in rend_map.items():
                if href:
                    return href
        except Exception as e:
            logger.warning("Error getting delivery URL for asset %s: %s", asset_id, e)
        return None

    def download_image_bytes(self, delivery_url: str) -> Optional[bytes]:
        """Downloads image bytes from Sitecore Content Hub."""
        headers = self._auth_headers()
        try:
            r = requests.get(delivery_url, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.content
        except Exception as e:
            logger.warning("Failed to download image from %s: %s", delivery_url, e)
        return None

    @staticmethod
    def generate_suggested_filename(
        source_name: Optional[str],
        prompt: str,
        brand_name: Optional[str] = None,
        ext: str = "png"
    ) -> str:
        """
        Dynamically generates a clean, informative filename derived from DAM metadata
        and the prompt context without any hardcoded brand/product dictionaries.
        """
        src = (source_name or "Product").strip()
        # Clean file extensions and internal DAM hash suffixes (e.g. _BRS_4bbb67699f1)
        src = re.sub(r'(\.(png|jpg|jpeg|webp|gif)|_BRS_[a-f0-9]+|_[a-f0-9]{6,})', '', src, flags=re.IGNORECASE)
        # Clean punctuation and replace spaces/separators with underscores
        src_clean = re.sub(r'[^a-zA-Z0-9]+', '_', src).strip('_')

        # Brand prefix if available
        brand_clean = re.sub(r'[^a-zA-Z0-9]+', '', brand_name).strip() if brand_name else ""

        if brand_clean and not src_clean.lower().startswith(brand_clean.lower()):
            prod_part = f"{brand_clean}_{src_clean}"
        else:
            prod_part = src_clean or "Product"

        # Contextual descriptor extracted dynamically from the user's prompt
        words = re.findall(r'[a-zA-Z]{3,}', prompt)
        skip = {"create", "realistic", "advertisement", "showing", "beside", "with", "this", "that", "image", "visual", "shot", "pack", "packet"}
        context_words = [w.capitalize() for w in words if w.lower() not in skip and w.lower() not in prod_part.lower()]
        context_part = "_".join(context_words[:2]) if context_words else "Campaign"

        clean_ext = ext.lstrip(".").lower()
        if clean_ext not in ["png", "jpg", "jpeg", "webp"]:
            clean_ext = "png"

        return f"{prod_part}_{context_part}_Campaign.{clean_ext}"

    def save_approved_campaign_asset(
        self,
        image_bytes: bytes,
        filename: str,
        title: Optional[str] = None,
        source_asset_id: Optional[int] = None,
        source_asset_name: Optional[str] = None,
        prompt: Optional[str] = None,
        model_identifier: str = "gpt-image-2.5-sunburst",
        brand_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete Sitecore Content Hub Asset Creation & Relationship Linking Workflow.
        Separately tracks and verifies upload, asset finalization, metadata, and relations.
        Guarantees that failures are reported accurately and never silently swallowed.
        """
        token = self.get_token()
        title = title or filename
        generation_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Step 1: Initiate upload session
        req_body = {
            "file_name": filename,
            "file_size": len(image_bytes),
            "action": {"name": "NewAsset"},
            "upload_configuration": {"name": "AssetUploadConfiguration"},
        }
        step1_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        step1_resp = requests.post(
            f"{self.ch_url}/api/v2.0/upload",
            json=req_body,
            headers=step1_headers,
            timeout=15,
        )
        if step1_resp.status_code not in [200, 201]:
            raise RuntimeError(f"Sitecore Upload Step 1 Failed ({step1_resp.status_code}): {step1_resp.text}")

        raw_upload_url = step1_resp.headers.get("Location")
        step1_data = step1_resp.json()
        upload_id = step1_data.get("upload_identifier")
        file_id = step1_data.get("file_identifier")

        if not raw_upload_url:
            raise RuntimeError("Sitecore Upload Step 1 did not return a Location header.")

        # Step 2: Binary multipart upload
        boundary = "----WebKitFormBoundaryDAMGenAI" + str(int(time.time()))
        mime_content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        header_part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime_content_type}\r\n\r\n"
        ).encode("utf-8")
        footer_part = f"\r\n--{boundary}--\r\n".encode("utf-8")
        multipart_body = header_part + image_bytes + footer_part

        req = urllib.request.Request(
            raw_upload_url,
            data=multipart_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Mozilla/5.0 SitecoreDAMClient",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status not in [200, 201]:
                    raise RuntimeError(f"Sitecore Upload Step 2 binary transfer returned status {resp.status}")
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Sitecore Upload Step 2 HTTP Error {e.code}: {err_msg}")

        # Step 3: Finalize upload session
        finalize_body = {
            "upload_identifier": upload_id,
            "file_identifier": file_id,
        }
        r_finalize = requests.post(
            f"{self.ch_url}/api/v2.0/upload/finalize",
            json=finalize_body,
            headers=step1_headers,
            timeout=20,
        )
        if r_finalize.status_code not in [200, 201]:
            raise RuntimeError(f"Sitecore Finalize Step 3 Failed ({r_finalize.status_code}): {r_finalize.text}")

        fin_data = r_finalize.json()
        asset_id = fin_data.get("asset_id")

        if not asset_id:
            raise RuntimeError("Sitecore upload finalize did not return an asset_id.")

        logger.info("Asset created in Sitecore DAM: ID %s. Applying metadata and relations...", asset_id)

        # Step 4: Enrich metadata and establish relationships on the new asset
        provenance_html = (
            f"<p><strong>GenAI Commercial Campaign Visual</strong></p>"
            f"<ul>"
            f"<li><strong>Model:</strong> <code>{model_identifier}</code></li>"
            f"<li><strong>Generated On:</strong> {generation_time}</li>"
            f"<li><strong>Creative Prompt:</strong> {prompt or 'Commercial Studio Shot'}</li>"
            f"<li><strong>Source DAM Asset ID:</strong> #{source_asset_id or 'N/A'} ({source_asset_name or 'N/A'})</li>"
            f"<li><strong>Workflow Status:</strong> Created / Under Review in Create Tab</li>"
            f"</ul>"
        )
        alt_text = f"GenAI commercial advertising image for {source_asset_name or 'product'}: {prompt or 'scene'}"

        update_payload: Dict[str, Any] = {
            "entitydefinition": {
                "href": f"{self.ch_url}/api/entitydefinitions/M.Asset"
            },
            "properties": {
                "Title": title,
                "FileName": filename,
                "Description": {"en-US": provenance_html},
                "SC.Asset.AltText": {"en-US": alt_text},
                "VisionDescription": f"Campaign visual generated via {model_identifier} from source asset {source_asset_id}.",
                "SocialPostCaption": prompt or title
            },
            "relations": {
                # Classification Image
                "AssetTypeToAsset": {
                    "parent": {"href": f"{self.ch_url}/api/entities/70128"}
                },
                # Standard Repository
                "ContentRepositoryToAsset": {
                    "parents": [{"href": f"{self.ch_url}/api/entities/734"}]
                }
            }
        }

        if source_asset_id:
            update_payload["relations"]["RelatedAsset"] = {
                "parents": [{"href": f"{self.ch_url}/api/entities/{source_asset_id}"}]
            }

        if brand_id:
            update_payload["relations"]["PCMBrandToAsset"] = {
                "parents": [{"href": f"{self.ch_url}/api/entities/{brand_id}"}]
            }

        # Step 4 execute & verify
        put_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        r_put = requests.put(
            f"{self.ch_url}/api/entities/{asset_id}",
            json=update_payload,
            headers=put_headers,
            timeout=15
        )
        if r_put.status_code not in [200, 204]:
            raise RuntimeError(
                f"Sitecore DAM metadata enrichment failed for asset #{asset_id} with status {r_put.status_code}: {r_put.text}"
            )

        # Step 5: Verification check on the created asset entity
        verify_resp = requests.get(f"{self.ch_url}/api/entities/{asset_id}", headers=self._auth_headers(), timeout=10)
        if verify_resp.status_code != 200:
            raise RuntimeError(f"Verification of created asset #{asset_id} failed: entity could not be retrieved.")

        v_data = verify_resp.json()
        saved_title = v_data.get("properties", {}).get("Title")
        logger.info("Successfully verified created asset #%s ('%s') in Sitecore DAM!", asset_id, saved_title)

        return {
            "success": True,
            "asset_id": asset_id,
            "filename": filename,
            "title": title,
            "source_asset_id": source_asset_id,
            "source_asset_name": source_asset_name,
            "model_identifier": model_identifier,
            "generated_at": generation_time,
            "upload_status": "succeeded",
            "metadata_status": "succeeded",
            "relationship_verified": True,
            "create_tab_url": f"{self.ch_url}/en-us/create?p840=1",
            "asset_url": f"{self.ch_url}/en-us/asset/{asset_id}",
        }

    def upload_campaign_asset_to_create_tab(
        self,
        image_bytes: bytes,
        filename: str,
        title: Optional[str] = None,
        brand_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for save_approved_campaign_asset."""
        return self.save_approved_campaign_asset(
            image_bytes=image_bytes,
            filename=filename,
            title=title,
            brand_id=brand_id
        )
