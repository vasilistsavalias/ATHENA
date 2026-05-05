import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import os
import json
import shutil

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.data_acquisition import DataAcquisition

class TestDataAcquisition(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests/temp_stage01")
        self.output_dir = self.base_dir / "raw"
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)

    @patch('thesis_pipeline.components.data_acquisition.requests.get')
    def test_download_flow(self, mock_get):
        """
        Verifies data integrity and filesystem operations.
        """
        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.__enter__.return_value = mock_search_resp 
        mock_search_resp.json.return_value = {
            "query": { "pages": { "101": {
                "title": "File:Artifact_01.jpg",
                "imageinfo": [{"url": "http://mock.url/image.jpg", "extmetadata": {}}]
            }}}
        }

        mock_img_resp = MagicMock()
        mock_img_resp.status_code = 200
        mock_img_resp.__enter__.return_value = mock_img_resp
        mock_img_resp.raw = MagicMock()
        mock_img_resp.raw.read.side_effect = [b"REAL_IMAGE_BYTES_123", b""] 

        def router(url, **kwargs):
            return mock_search_resp if "api.php" in url else mock_img_resp
        
        mock_get.side_effect = router

        acquirer = DataAcquisition(api_url="http://mock.api/api.php")
        
        with patch.object(DataAcquisition, '_check_image_quality', return_value=(True, "OK")):
            with patch.object(DataAcquisition, '_get_file_hash', return_value="hash_abc"):
                acquirer.download_images_from_category("Category:Vases", self.output_dir, limit=1)

        # DEBUG: List files if none found
        created_files = list(self.output_dir.glob("*.jpg*"))
        self.assertGreater(len(created_files), 0, f"No files created in {self.output_dir}")
        
        target_file = created_files[0]
        with open(target_file, "rb") as f:
            self.assertEqual(f.read(), b"REAL_IMAGE_BYTES_123")

        self.assertTrue((self.output_dir / "metadata" / f"{target_file.stem}.json").exists())

if __name__ == '__main__':
    unittest.main()