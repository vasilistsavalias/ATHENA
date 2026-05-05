import requests
import logging
import json
import time
import shutil
import re
import hashlib
from pathlib import Path
from PIL import Image, ImageStat
from tqdm import tqdm

class DataAcquisition:
    def __init__(self, api_url="https://commons.wikimedia.org/w/api.php"):
        self.api_url = api_url
        self.logger = logging.getLogger(__name__)
        self.headers = {
            "User-Agent": "AthenaResearchBot/1.0 (https://github.com/athena-project) based-on-requests/2.31"
        }

    def download_images_from_category(
        self,
        start_category,
        output_dir,
        limit=2500,
        *,
        filename_prefix: str = "wiki",
        state_file_name: str | None = None,
    ):
        """
        Populates output_dir using the Wikimedia Search API logic from V4 Scraper.
        """
        output_dir = Path(output_dir)
        metadata_dir = output_dir / "metadata"
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load State / History
        state_file = output_dir / (state_file_name or f"scraper_state_{filename_prefix}.json")
        state = {"offset": 0, "downloaded_hashes": [], "downloaded_titles": []}
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

        # 2. Build Cache from existing files if state is empty
        existing_files = (
            list(output_dir.glob(f"{filename_prefix}_*.jpg"))
            + list(output_dir.glob(f"{filename_prefix}_*.jpeg"))
            + list(output_dir.glob(f"{filename_prefix}_*.png"))
        )
        current_count = len(existing_files)
        
        if not state["downloaded_hashes"] and current_count > 0:
            self.logger.info("Rebuilding hash cache from existing files...")
            for p in existing_files:
                if p.is_file():
                    state["downloaded_hashes"].append(self._get_file_hash(p))
            state["downloaded_titles"] = [p.name for p in existing_files]

        if current_count >= limit:
            self.logger.info(f"Target of {limit} images already met. Skipping download.")
            return

        # 3. Main Loop
        offset = state.get("offset", 0)
        # Search API prefers spaces over underscores
        search_query = start_category.replace("Category:", "").replace("_", " ")
        
        self.logger.info(f"Starting acquisition. Query: '{search_query}'. Target: {limit}. Current: {current_count}")

        while current_count < limit:
            data = self._search_images(search_query, offset=offset)
            if not data or "query" not in data:
                self.logger.warning("No more results from API.")
                break
                
            pages = data["query"].get("pages", {})
            if not pages:
                break
            
            self.logger.info(f"Processing batch at offset {offset} ({len(pages)} candidates)...")
                
            for page_id, page in pages.items():
                if current_count >= limit: break
                
                title = page["title"]
                if title in state["downloaded_titles"]: continue
                if "imageinfo" not in page: continue
                
                info = page["imageinfo"][0]
                url = info["url"]
                
                # Safe Filename
                safe_title = re.sub(r'[\\/*?:"<>|]', "", title.replace("File:", ""))[:100]
                filename = output_dir / f"{filename_prefix}_{current_count:05d}_{safe_title}.jpg"
                json_filename = metadata_dir / f"{filename_prefix}_{current_count:05d}_{safe_title}.json"
                
                self.logger.info(f"  [{current_count}/{limit}] Downloading: {title[:50]}...")
                if self._download_file(url, filename):
                    f_hash = self._get_file_hash(filename)
                    if f_hash in state["downloaded_hashes"]:
                        filename.unlink()
                    else:
                        is_valid, reason = self._check_image_quality(filename)
                        if is_valid:
                            extmeta = info.get("extmetadata", {}) or {}
                            # Keep legacy key "metadata" for downstream compatibility (Stage 06).
                            meta = {
                                "source": "wikimedia",
                                "source_id": title,
                                "source_page_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                                "download_url": url,
                                "license": (
                                    (extmeta.get("LicenseShortName") or {}).get("value")
                                    or (extmeta.get("License") or {}).get("value")
                                    or ""
                                ),
                                "title": title,
                                "url": url,
                                "metadata": extmeta,
                                "raw_metadata": info,
                            }
                            with open(json_filename, "w") as f:
                                json.dump(meta, f, indent=2)
                            
                            state["downloaded_hashes"].append(f_hash)
                            state["downloaded_titles"].append(title)
                            current_count += 1
                        else:
                            filename.unlink()
                
                time.sleep(0.1)

            # Update State
            offset = data.get("continue", {}).get("gsroffset", offset + 50)
            state["offset"] = offset
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

        self.logger.info(f"Acquisition finished. Total images: {current_count}")

    def _get_file_hash(self, path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _check_image_quality(self, img_path):
        try:
            with Image.open(img_path) as img:
                if img.width < 300 or img.height < 300: return False, "LowRes"
                if img.mode == 'L': return False, "Grayscale"
                img_hsv = img.convert('HSV')
                stat = ImageStat.Stat(img_hsv)
                if stat.mean[1] < 15: return False, "LowSat"
                return True, "OK"
        except: return False, "Error"

    def _search_images(self, query, limit=50, offset=0):
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
            "gsrlimit": limit, "gsroffset": offset,
            "prop": "imageinfo", "iiprop": "url|extmetadata|size"
        }
        for attempt in range(3):
            try:
                r = requests.get(self.api_url, headers=self.headers, params=params, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    self.logger.warning(f"Rate-limited (429). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    self.logger.warning(f"API returned status {r.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                wait = 2 ** (attempt + 1)
                self.logger.warning(f"Request error (attempt {attempt+1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return None

    def _download_file(self, url, filename):
        for attempt in range(3):
            try:
                with requests.get(url, headers=self.headers, stream=True, timeout=30) as r:
                    if r.status_code == 200:
                        with open(filename, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                        return True
                    elif r.status_code == 429:
                        time.sleep(2 ** (attempt + 1))
            except requests.exceptions.RequestException:
                time.sleep(2 ** (attempt + 1))
        return False
