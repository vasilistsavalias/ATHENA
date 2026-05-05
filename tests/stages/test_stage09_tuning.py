import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from box import ConfigBox
import os
import pytest
import sys
import matplotlib.pyplot as plt

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
import sys
import os
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from thesis_pipeline.components.model_training import ModelTrainer

class TestStage07(unittest.TestCase):
    def setUp(self):
        self.config = ConfigBox({
            "num_epochs": 1,
            "train_batch_size": 2,
            "save_model_epochs": 1,
            "output_dir": "tests/temp_models"
        })
        self.hyperparams = ConfigBox({
            "model_id": "stabilityai/stable-diffusion-2-inpainting",
            "learning_rate": 1e-5,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 1e-2,
            "adam_epsilon": 1e-08,
            "mixed_precision": "no"
        })
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        plt.close('all')
        if Path(self.config.output_dir).exists():
            shutil.rmtree(self.config.output_dir, ignore_errors=True)

    @patch('thesis_pipeline.components.model_training.Accelerator')
    @patch('thesis_pipeline.components.model_training.DDPMScheduler')
    @patch('thesis_pipeline.components.model_training.UNet2DConditionModel')
    @patch('thesis_pipeline.components.model_training.AutoencoderKL')
    @patch('thesis_pipeline.components.model_training.CLIPTextModel')
    @patch('thesis_pipeline.components.model_training.CLIPTokenizer')
    def test_train_loop(self, mock_tokenizer, mock_text_encoder, mock_vae, mock_unet, mock_scheduler, mock_accelerator):
        # Mock Accelerator
        accelerator = mock_accelerator.return_value
        accelerator.device = "cpu"
        # Prepare returns the inputs
        accelerator.prepare.side_effect = lambda *args: args
        # Unwrap model
        accelerator.unwrap_model.return_value = MagicMock()
        
        # Mock Gather/Mean/Item for Validation Loss Logging
        mock_gathered_loss = MagicMock()
        mock_gathered_loss.mean.return_value.item.return_value = 0.5 # Return a float for f-string formatting
        accelerator.gather.return_value = mock_gathered_loss

        # Mock VAE output
        vae_dist = MagicMock()
        vae_dist.sample.return_value = torch.randn(2, 4, 64, 64)
        mock_vae.from_pretrained.return_value.encode.return_value.latent_dist = vae_dist
        mock_vae.from_pretrained.return_value.config.scaling_factor = 0.18215
        
        # Mock Tokenizer
        tokenizer_instance = mock_tokenizer.from_pretrained.return_value
        tokenizer_instance.model_max_length = 77

        def _fake_tokenizer_call(text, *args, **kwargs):
            batch = len(text) if isinstance(text, list) else 1
            ids = torch.randint(0, 1000, (batch, 77))
            enc = MagicMock()
            enc.input_ids = ids
            enc.__getitem__.side_effect = lambda k: {"input_ids": ids}[k]
            return enc

        tokenizer_instance.side_effect = _fake_tokenizer_call
        tokenizer_instance.decode.return_value = "decoded prompt"
        
        # Mock Text Encoder
        mock_text_encoder.from_pretrained.return_value.return_value = [torch.randn(2, 77, 768)] # hidden states
        
        # Mock Scheduler
        scheduler = mock_scheduler.from_pretrained.return_value
        scheduler.config.num_train_timesteps = 1000
        scheduler.add_noise.return_value = torch.randn(2, 4, 64, 64)
        
        # Mock UNet structure
        # instance = UNet(...)
        # output = instance(inputs)
        # prediction = output.sample
        
        unet_instance = MagicMock()
        unet_instance.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
        
        unet_output_obj = MagicMock()
        unet_output_obj.sample = torch.randn(2, 4, 64, 64)
        
        unet_instance.return_value = unet_output_obj
        
        mock_unet.from_pretrained.return_value = unet_instance
        
        # Mock Batch with Caption
        batch = {
            "original_image": torch.randn(2, 3, 512, 512),
            "masked_image": torch.randn(2, 3, 512, 512),
            "mask": torch.randn(2, 1, 512, 512),
            "caption": ["A red vase", "A black vase"]
        }
        
        train_dataloader = [batch]
        val_dataloader = [batch]
        
        output_dir = Path(self.config.output_dir)
        trainer = ModelTrainer(self.config, self.hyperparams, output_dir=output_dir, report_dir=output_dir, accelerator=accelerator)
        trainer.train(train_dataloader, val_dataloader)
        
        # Verification (lightweight): training loop executed and tokenizer was exercised.
        self.assertGreaterEqual(tokenizer_instance.call_count, 1)

if __name__ == '__main__':
    unittest.main()
