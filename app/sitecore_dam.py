import os
import io
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
        self.ch_url = (ch_url or os.getenv("SITECORE_CH_URL", "")).rstrip("/")
        self.username = username or os.getenv("SITECORE_USERNAME", "")
        self.password = password or os.getenv("SITECORE_PASSWORD", "")
        self.client_id = client_id or os.getenv("SITECORE_CLIENT_ID", "ImageUpload")
        self.client_secret = client_secret or os.getenv("SITECORE_CLIENT_SECRET", "")
        
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

    def search_brand_and_product(self, prompt: str) -> Dict[str, Any]:
        """
        Dynamically analyzes user prompt and retrieves matching brand details,
        approved product packshot/render, and official brand logo from Sitecore DAM.
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
            "matched_keywords": [],
        }

        # 1. Query Brands from Sitecore Content Hub
        brands_url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Brand'&take=20"
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
                        result["matched_keywords"].append(bname)
                        break
                
                # If no brand explicitly mentioned, check if "good day" implies Britannia or default to first
                if not result["brand_name"]:
                    if any(w in prompt_lower for w in ["good day", "goodday", "biscuit", "cookie", "rusk", "bread", "cheese", "jim jam", "bourbon"]):
                        result["brand_id"] = 70326
                        result["brand_name"] = "Britannia"
                        result["brand_tagline"] = "EXCITING GOODNESS IN EVERY BITE"
                        result["matched_keywords"].append("Britannia (inferred)")
                    elif len(brands) > 0:
                        b = brands[0]
                        result["brand_id"] = b.get("id")
                        result["brand_name"] = b.get("properties", {}).get("BrandName", "Brand")
        except Exception as e:
            logger.warning(f"Error querying M.Brand: {e}")

        # 2. Query Assets from DAM to find Product and Official Logo
        assets_query_url = f"{self.ch_url}/api/entities/query?query=Definition.Name=='M.Asset'&take=100"
        try:
            r_assets = requests.get(assets_query_url, headers=headers, timeout=12)
            if r_assets.status_code == 200:
                assets = r_assets.json().get("items", [])
                bname_lower = (result["brand_name"] or "").lower()
                
                # A. Find Official Logo
                best_logo = None
                for a in assets:
                    props = a.get("properties", {})
                    title = (props.get("Title") or "").lower()
                    fname = (props.get("FileName") or "").lower()
                    
                    if "logo" in title or "logo" in fname:
                        # Prioritize logo matching brand name (e.g. britannia_logo)
                        if bname_lower and (bname_lower in title or bname_lower in fname):
                            best_logo = a
                            break
                        elif not best_logo:
                            best_logo = a
                            
                if best_logo:
                    result["brand_logo_url"] = self._get_asset_delivery_url(best_logo.get("id"), preferred=["preview", "downloadPreview", "bigthumbnail", "downloadOriginal"])
                    logger.info(f"Found official brand logo in DAM: Asset ID {best_logo.get('id')}")

                # B. Find Matching Product Packshot / Render
                best_match = None
                best_score = 0
                ignore = {"a", "an", "the", "in", "on", "at", "for", "with", "and", "or", "of", "to", "scene", "campaign", "image", "create", "visual", "photo"}
                prompt_words = [w.strip(".,!?") for w in prompt_lower.split() if w.strip(".,!?") not in ignore and len(w) > 2]
                
                for a in assets:
                    props = a.get("properties", {})
                    title = (props.get("Title") or "").lower()
                    fname = (props.get("FileName") or "").lower()
                    combined = f"{title} {fname}"
                    
                    # Skip archives/media/logos
                    if any(combined.endswith(ext) for ext in [".zip", ".pdf", ".mp4"]) or "logo" in combined:
                        continue
                        
                    score = 0
                    for pw in prompt_words:
                        if pw in combined:
                            score += 1
                            
                    # Special boost for phrase matches like "good day" or "goodday"
                    if "good day" in prompt_lower or "goodday" in prompt_lower:
                        if "goodday" in combined or "good day" in combined:
                            score += 10
                            
                    if score > best_score:
                        best_score = score
                        best_match = a
                        
                if best_match and best_score > 0:
                    result["product_id"] = best_match.get("id")
                    result["product_name"] = best_match.get("properties", {}).get("Title")
                    result["product_image_url"] = self._get_asset_delivery_url(best_match.get("id"), preferred=["downloadOriginal", "downloadPreview", "preview"])
                    result["matched_keywords"].append(f"Product: {result['product_name']}")
                    logger.info(f"Found product asset in DAM: Asset ID {result['product_id']} ({result['product_name']})")
        except Exception as e:
            logger.warning(f"Error querying DAM assets: {e}")

        return result

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

