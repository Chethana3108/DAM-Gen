import os
import io
import re
import time
import logging
import urllib.request
import urllib.parse
import requests
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SitecoreDAMClient:
    """
    Universal Sitecore Content Hub DAM Client.
    Handles dynamic brand/product asset retrieval and direct asset ingestion
    into the Sitecore Content Hub 'Create' sandbox (/en-us/create?p840=1).
    """

    def __init__(
        self,
        ch_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.ch_url = (ch_url or os.getenv("SITECORE_CH_URL", "https://btr-q-002.sitecorecontenthub.cloud")).rstrip("/")
        self.username = username or os.getenv("SITECORE_USERNAME", "kalpesh@biztechnosys.com")
        self.password = password or os.getenv("SITECORE_PASSWORD", "P@ssw0rd7932")
        self.client_id = client_id or os.getenv("SITECORE_CLIENT_ID", "ImageUpload")
        self.client_secret = client_secret or os.getenv("SITECORE_CLIENT_SECRET", "URXfOKv-j-GoHXp0KhiRhQS9oEpm1uNlFE_ZXLrb2qg")
        
        self._token: Optional[str] = None
        self._token_expiry: float = 0

    def get_token(self, force_refresh: bool = False) -> str:
        """Obtain or reuse a valid OAuth Bearer token."""
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
            logger.error(f"Failed to authenticate with Sitecore Content Hub: {e}")
            raise RuntimeError(f"Sitecore Content Hub Authentication Failed: {e}")

    def _auth_headers(self) -> Dict[str, str]:
        token = self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def search_brand_and_product(self, prompt: str, exact_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Dynamically analyzes user prompt or exact filename and retrieves matching brand details,
        approved product asset, all linked layers (renditions, related sub-assets),
        and official brand logo from Sitecore DAM.
        Includes disambiguation if multiple assets match with similar confidence.
        """
        token = self.get_token()
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
            logger.warning(f"Error querying M.Brand: {e}")

        # 2. Query Assets from DAM
        assets = []
        assets_query_url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Asset'&take=100"
        try:
            r_assets = requests.get(assets_query_url, headers=headers, timeout=12)
            if r_assets.status_code == 200:
                assets = r_assets.json().get("items", [])
        except Exception as e:
            logger.warning(f"Error querying DAM assets: {e}")

        valid_image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

        # Step A: If exact filename was supplied (or detected in prompt)
        if exact_filename:
            clean_exact = exact_filename.strip().lower()
            fname_stem = re.sub(r'\.(png|jpg|jpeg|webp|gif)$', '', clean_exact)
            fname_tokens = [tok for tok in re.split(r'[_\-\s\.]+', fname_stem) if len(tok) > 2 and tok not in ["png", "jpg", "jpeg", "webp"]]
            exact_matches = []
            
            for a in assets:
                props = a.get("properties", {})
                fname = (props.get("FileName") or "").lower()
                title = (props.get("Title") or "").lower()
                combined_name = f"{title} {fname}"

                if not any(fname.endswith(ext) for ext in valid_image_exts):
                    continue

                # 1. Exact string containment
                if clean_exact in fname or fname in clean_exact or fname_stem in fname:
                    exact_matches.append(a)
                # 2. Token set containment (e.g. "Britannia_JimJam.png" matches "Jim_Jam_BRS...png")
                elif fname_tokens and all(tok in combined_name or tok in combined_name.replace('_', '').replace(' ', '') for tok in fname_tokens if tok != "britannia"):
                    exact_matches.append(a)

            if len(exact_matches) > 1:
                # Multiple assets share this name; require user disambiguation
                result["requires_disambiguation"] = True
                result["candidates"] = [
                    {"id": m.get("id"), "title": m.get("properties", {}).get("Title"), "filename": m.get("properties", {}).get("FileName")}
                    for m in exact_matches[:5]
                ]
                logger.info(f"Multiple DAM assets matched '{exact_filename}'. Disambiguation required.")
                return result
            elif len(exact_matches) == 1:
                best_match = exact_matches[0]
                logger.info(f"Exact match found for '{exact_filename}': Asset ID {best_match.get('id')}")
                return self._populate_asset_details(best_match, result, brands, headers)

        # Step B: Keyword and token search
        context_words = {"table", "cup", "coffee", "tea", "desk", "counter", "surface", "background", "hand", "hands", "holding", "hold", "person", "celebrity", "woman", "man", "model", "morning", "night", "light", "sunlight", "outdoor", "indoor"}
        stop_words = {"a", "an", "the", "in", "on", "at", "for", "with", "and", "or", "of", "to", "is", "like", "by", "scene", "campaign", "image", "create", "visual", "photo", "this", "that", "want", "you", "generate", "xyz", "particular", "some", "made", "placed", "where", "png", "jpg", "jpeg"}
        all_ignore = stop_words | context_words

        # Tokenize by space, underscore, dash, and dot
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

            if not any(fname.endswith(ext) for ext in valid_image_exts) or "logo" in combined:
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

        # Fallback to first valid image asset if no keyword match
        if not best_match and assets:
            for a in assets:
                fname = (a.get("properties", {}).get("FileName") or "").lower()
                if any(fname.endswith(ext) for ext in valid_image_exts) and "logo" not in fname:
                    best_match = a
                    break

        if best_match:
            return self._populate_asset_details(best_match, result, brands, headers, candidate_assets=candidate_assets, all_ignore=all_ignore)

        return result

    def _populate_asset_details(self, best_match: Dict[str, Any], result: Dict[str, Any], brands: list, headers: dict, candidate_assets: list = None, all_ignore: set = None) -> Dict[str, Any]:
        """Populates rich asset metadata, linked layers, and official logo."""
        all_ignore = all_ignore or set()
        candidate_assets = candidate_assets or []
        pid = best_match.get("id")
        props = best_match.get("properties", {})
        result["product_id"] = pid
        result["product_name"] = props.get("Title") or props.get("FileName")
        result["product_image_url"] = self._get_asset_delivery_url(pid, preferred=["downloadOriginal", "vision_ready", "preview", "downloadPreview"])
        
        desc_dict = props.get("Description") or {}
        result["description"] = desc_dict.get("en-US", "") if isinstance(desc_dict, dict) else str(desc_dict)
        alt_dict = props.get("SC.Asset.AltText") or {}
        result["alt_text"] = alt_dict.get("en-US", "") if isinstance(alt_dict, dict) else str(alt_dict)
        
        ai_sugg = props.get("AI.Suggestions") or {}
        tag_suggestions = ai_sugg.get("TagToAsset", []) if isinstance(ai_sugg, dict) else []
        sim_tags = [t.get("name") for t in (props.get("ImageSimilarityTags") or []) if isinstance(t, dict) and "name" in t]
        result["tags"] = list(set(tag_suggestions + sim_tags))
        result["matched_keywords"].append(f"Product: {result['product_name']}")
        logger.info(f"Dynamically resolved product asset in DAM: ID {pid} ('{result['product_name']}')")

        # Dynamic Linked Layer Discovery
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

                # If Brand wasn't identified from prompt, infer from asset relations
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
            logger.warning(f"Error inspecting entity relations for asset {pid}: {e}")

        # Sibling layer assets sharing the same product name root
        clean_pname = re.sub(r'[_\-\s]+', ' ', (result["product_name"] or "").lower())
        pname_tokens = [tok for tok in clean_pname.split() if len(tok) > 3 and tok not in all_ignore][:2]
        if pname_tokens:
            for score, cand in candidate_assets:
                cid = cand.get("id")
                if cid != pid and cid not in layer_ids:
                    c_title = (cand.get("properties", {}).get("Title") or "").lower()
                    c_file = (cand.get("properties", {}).get("FileName") or "").lower()
                    if any(tok in c_title or tok in c_file for tok in pname_tokens):
                        layer_ids.add(cid)
                        logger.info(f"Discovered linked sibling layer asset: ID {cid} ('{c_title}')")

        for lid in layer_ids:
            l_url = self._get_asset_delivery_url(lid, preferred=["downloadOriginal", "vision_ready", "preview", "downloadPreview"])
            if l_url and l_url not in result["layer_urls"]:
                result["layer_urls"].append(l_url)
                logger.info(f"Attached linked layer from DAM: Asset ID {lid}")

        # Check renditions on the primary asset for alternative layer files
        try:
            rend_url = f"{self.ch_url}/api/entities/{pid}/renditions"
            r_rend = requests.get(rend_url, headers=headers, timeout=10)
            if r_rend.status_code == 200:
                for ritem in r_rend.json().get("renditions", []):
                    rname = (ritem.get("rendition_link", {}).get("name") or "").lower()
                    if any(k in rname for k in ["alternative", "transparent", "layer", "cutout"]):
                        files = ritem.get("file_location", {}).get("files", [])
                        if files and "delivery_link" in files[0]:
                            l_href = files[0]["delivery_link"]["href"]
                            if l_href not in result["layer_urls"]:
                                result["layer_urls"].append(l_href)
                                logger.info(f"Attached linked layer rendition '{rname}' from Asset {pid}")
        except Exception as e:
            logger.warning(f"Error checking layer renditions: {e}")

        # Dynamic Brand Logo Discovery
        bname_lower = (result["brand_name"] or "").lower()
        assets_query_url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Asset'&take=50"
        try:
            r_a = requests.get(assets_query_url, headers=headers, timeout=10)
            if r_a.status_code == 200:
                for a in r_a.json().get("items", []):
                    props = a.get("properties", {})
                    combined = f"{props.get('Title', '')} {props.get('FileName', '')}".lower()
                    if "logo" in combined:
                        if bname_lower and bname_lower in combined:
                            result["brand_logo_url"] = self._get_asset_delivery_url(a.get("id"))
                            break
                        elif not result["brand_logo_url"]:
                            result["brand_logo_url"] = self._get_asset_delivery_url(a.get("id"))
        except Exception as e:
            logger.warning(f"Error finding logo: {e}")

        return result

    def get_asset_by_id(self, asset_id: int) -> Dict[str, Any]:
        """Directly retrieves an asset and its linked layers by ID (used for disambiguation selection)."""
        headers = self._auth_headers()
        ent_url = f"{self.ch_url}/api/entities/{asset_id}"
        resp = requests.get(ent_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise ValueError(f"Asset ID {asset_id} not found in DAM.")
        asset_data = resp.json()
        result = {
            "brand_id": None, "brand_name": None, "brand_tagline": None,
            "brand_logo_url": None, "product_id": asset_id, "product_name": None,
            "product_image_url": None, "description": "", "alt_text": "", "tags": [],
            "layer_urls": [], "matched_keywords": [], "requires_disambiguation": False, "candidates": []
        }
        return self._populate_asset_details(asset_data, result, [], headers)

    def _get_asset_delivery_url(self, asset_id: int, preferred: list = None) -> Optional[str]:
        """Resolves the direct authenticated delivery URL for an asset's preferred rendition."""
        preferred = preferred or ["preview", "downloadPreview", "downloadOriginal", "bigthumbnail"]
        headers = self._auth_headers()
        url = f"{self.ch_url}/api/entities/{asset_id}/renditions"
        
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                return None
            renditions = r.json().get("renditions", [])
            
            # Map by rendition name
            rend_map = {}
            for ritem in renditions:
                rname = ritem.get("rendition_link", {}).get("name")
                files = ritem.get("file_location", {}).get("files", [])
                if files and "delivery_link" in files[0]:
                    rend_map[rname] = files[0]["delivery_link"]["href"]
                    
            for pref in preferred:
                if pref in rend_map:
                    return rend_map[pref]
                    
            # Fallback to any available delivery link
            for k, href in rend_map.items():
                if href:
                    return href
        except Exception as e:
            logger.warning(f"Error getting delivery URL for asset {asset_id}: {e}")
        return None

    def download_image_bytes(self, delivery_url: str) -> Optional[bytes]:
        """Downloads image bytes from Sitecore Content Hub."""
        headers = self._auth_headers()
        try:
            r = requests.get(delivery_url, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.content
        except Exception as e:
            logger.warning(f"Failed to download image from {delivery_url}: {e}")
        return None

    def upload_campaign_asset_to_create_tab(
        self,
        image_bytes: bytes,
        filename: str,
        title: Optional[str] = None,
        brand_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Uploads a generated campaign image directly into Sitecore Content Hub DAM
        as a new M.Asset under the 'Create' sandbox tab (/en-us/create?p840=1).
        """
        token = self.get_token()
        title = title or filename
        
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

        # Step 2: Binary Upload preserving raw URL encoding
        boundary = "----WebKitFormBoundaryDAMGenAI" + str(int(time.time()))
        header_part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
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

        # Step 3: Finalize upload
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

        return {
            "success": True,
            "asset_id": asset_id,
            "filename": filename,
            "title": title,
            "create_tab_url": f"{self.ch_url}/en-us/create?p840=1",
            "asset_url": f"{self.ch_url}/en-us/asset/{asset_id}" if asset_id else None,
        }

