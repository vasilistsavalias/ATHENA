import requests
import logging
import json
import time
import shutil
import re
import hashlib
from pathlib import Path
from PIL import Image, ImageStat

# --- Configuration ---
OUTPUT_DIR = Path("data/01_raw/wikimedia_collection")
METADATA_DIR = OUTPUT_DIR / "metadata"
STATE_FILE = OUTPUT_DIR / "scraper_state.json"
TARGET_COUNT = 50000  # Fetch maximum available from Wikimedia Commons
SEARCH_QUERY = "Ancient Greek pottery"
BASE_URL = "https://commons.wikimedia.org/w/api.php"

# Logging
LOG_DIR = Path("outputs/00_logs/scraper")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "wikimedia_v4.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "AthenaResearchBot/1.0 (https://github.com/athena-project) based-on-requests/2.31"
}

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"offset": 0, "downloaded_hashes": [], "downloaded_titles": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_image_quality(img_path):
    try:
        with Image.open(img_path) as img:
            if img.width < 300 or img.height < 300:
                return False, f"Low Res ({img.size})"
            if img.mode == 'L':
                return False, "Grayscale"
            img = img.convert('HSV')
            stat = ImageStat.Stat(img)
            mean_sat = stat.mean[1]
            if mean_sat < 15: # Slightly relaxed saturation check
                return False, f"Low Saturation ({mean_sat:.2f})"
            return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"

def search_images(query, limit=50, offset=0):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size"
    }
    try:
        r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        logger.error(f"Request Exception: {e}")
        return None

def download_file(url, filename):
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=30) as r:
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                return True
    except Exception:
        return False
    return False

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    state = load_state()
    offset = state["offset"]
    
    # Initialize seen hashes from existing files if state is empty but files exist
    existing_files = list(OUTPUT_DIR.glob("*.jpg"))
    current_count = len(existing_files)
    
    if not state["downloaded_hashes"] and current_count > 0:
        logger.info("Building hash cache from existing files...")
        for p in existing_files:
            state["downloaded_hashes"].append(get_file_hash(p))
            state["downloaded_titles"].append(p.name) # Approximate
        save_state(state)

    logger.info(f"Starting Scraper V4. Target: {TARGET_COUNT}. Current: {current_count}. Offset: {offset}")

    while current_count < TARGET_COUNT:
        data = search_images(SEARCH_QUERY, limit=50, offset=offset)
        
        if not data or "query" not in data:
            logger.warning("No more results or API failure.")
            break
            
        pages = data["query"]["pages"]
        if not pages:
            break
            
        logger.info(f"Processing batch at offset {offset} ({len(pages)} items)...")
        
        batch_processed = 0
        for page_id, page in pages.items():
            if current_count >= TARGET_COUNT: break
            
            title = page["title"]
            if title in state["downloaded_titles"]:
                continue

            if "imageinfo" not in page: continue
            info = page["imageinfo"][0]
            url = info["url"]
            
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title.replace("File:", ""))[:120]
            # Use a unique ID to prevent overwrites, but relying on title check mostly
            filename = OUTPUT_DIR / f"wiki_{current_count:05d}_{safe_title}.jpg"
            json_filename = METADATA_DIR / f"wiki_{current_count:05d}_{safe_title}.json"
            
            # Download
            if download_file(url, filename):
                # Check Hash (Deduplication)
                f_hash = get_file_hash(filename)
                if f_hash in state["downloaded_hashes"]:
                    logger.info(f"[-] Duplicate content: {title}")
                    filename.unlink()
                else:
                    # Quality Gate
                    is_valid, reason = check_image_quality(filename)
                    if is_valid:
                        # Save Meta
                        meta = {"title": title, "url": url, "metadata": info.get("extmetadata", {})}
                        with open(json_filename, "w") as f:
                            json.dump(meta, f, indent=2)
                        
                        logger.info(f"[+] Saved: {safe_title[:30]}...")
                        state["downloaded_hashes"].append(f_hash)
                        state["downloaded_titles"].append(title)
                        current_count += 1
                        batch_processed += 1
                    else:
                        logger.info(f"[-] Filtered ({reason}): {title}")
                        filename.unlink()
            
            time.sleep(0.2)
            
        # Update Offset
        if "continue" in data:
            offset = data["continue"]["gsroffset"]
        else:
            offset += 50 # Fallback
            
        state["offset"] = offset
        save_state(state)
        
        if batch_processed == 0 and len(pages) > 0:
             logger.warning("Batch yielded 0 new images (mostly duplicates). Moving to next batch.")

if __name__ == "__main__":
    main()
