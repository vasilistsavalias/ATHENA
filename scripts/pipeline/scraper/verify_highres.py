import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.carc.ox.ac.uk"
SAMPLE_THUMB = "/Vases/SPIFF/Images200/GER37/CVA.GER37.1830.2/ac001001.jpe"

VARIANTS = [
    "/Vases/SPIFF/Images/{path}",        # Common pattern
    "/Vases/SPIFF/Images1000/{path}",    # Larger resize?
    "/Vases/Images/{path}",              # Root images
    "/Vases/ASP/Images/{path}",          # Another guess
    "/Vases/SPIFF/Images500/{path}",     # Medium?
]

def verify_highres():
    # Extract the relative path after Images200
    # Pattern: /Vases/SPIFF/Images200/(.*)
    rel_path = SAMPLE_THUMB.split("Images200/")[1]
    
    logger.info(f"Base Thumb: {BASE_URL}{SAMPLE_THUMB}")
    
    for fmt in VARIANTS:
        test_path = fmt.format(path=rel_path)
        url = BASE_URL + test_path
        try:
            r = requests.head(url, timeout=5)
            if r.status_code == 200:
                logger.info(f"[FOUND] {url} (Content-Length: {r.headers.get('Content-Length')})")
            else:
                logger.warning(f"[MISSING] {url} ({r.status_code})")
        except Exception as e:
            logger.error(f"Error checking {url}: {e}")

if __name__ == "__main__":
    verify_highres()
