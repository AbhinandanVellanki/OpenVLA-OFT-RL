"""
Test suite for OpenVLA-OFT model loading and inference.

Run tests with: pytest test_oft_loading.py
Run with verbose output: pytest test_oft_loading.py -v
Run only fast tests: pytest test_oft_loading.py -m "not slow"
Run only slow/integration tests: pytest test_oft_loading.py -m slow
"""

import pytest
import torch
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

from load_vla import OpenVLAOFTModel, load_openvla_oft


# Test fixtures
@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.dtype = torch.float32
    return model


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing"""
    processor = Mock()
    return processor


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing"""
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


# Unit tests for load_openvla_oft function
class TestLoadOpenVLAOFT:
    """Test suite for load_openvla_oft function"""
    
    def test_custom_device(self, mock_model, mock_processor):
        """Test loading with custom device"""
        with patch('transformers.AutoProcessor.from_pretrained', return_value=mock_processor):
            with patch('transformers.AutoModelForVision2Seq.from_pretrained', return_value=mock_model):
                with patch('pathlib.Path.exists', return_value=False):
                    model, processor = load_openvla_oft("test-model-id", device="cpu")
                    mock_model.to.assert_called_once_with("cpu")
    

# Unit tests for OpenVLAOFTModel class
class TestOpenVLAOFTModel:
    """Test suite for OpenVLAOFTModel class"""
    
    def test_init_loads_model_and_processor(self, mock_model, mock_processor):
        """Test that __init__ loads model and processor"""
        with patch('load_vla.load_openvla_oft', return_value=(mock_model, mock_processor)):
            with patch('torch.cuda.is_available', return_value=False):
                model_wrapper = OpenVLAOFTModel("test-model-id")
                assert model_wrapper.model == mock_model
                assert model_wrapper.processor == mock_processor
    
    def test_forward_processes_inputs(self, mock_model, mock_processor, sample_image):
        """Test that forward method processes inputs correctly"""
        # Mock processor output
        mock_inputs = Mock()
        mock_inputs.to = Mock(return_value=mock_inputs)
        mock_processor.return_value = mock_inputs
        
        # Mock model output
        mock_output = Mock()
        mock_model.return_value = mock_output
        mock_model.dtype = torch.float32
        
        with patch('load_vla.load_openvla_oft', return_value=(mock_model, mock_processor)):
            with patch('torch.cuda.is_available', return_value=False):
                model_wrapper = OpenVLAOFTModel("test-model-id", device="cpu")
                
                # Call forward
                with patch('torch.no_grad'):
                    outputs = model_wrapper.forward("test prompt", sample_image)
                
                # Verify processor was called with correct arguments
                mock_processor.assert_called_once()
                call_args = mock_processor.call_args
                assert call_args[0][0] == "test prompt"  # prompt is first positional arg
                assert len(call_args[0][1]) == 1  # images list has one image
                # Verify model was called
                mock_model.assert_called_once()
                assert outputs == mock_output

# Integration tests (marked as slow because they require actual model loading)
@pytest.mark.slow
class TestOpenVLAOFTIntegration:
    """Integration tests that require actual model loading"""
    
    @pytest.mark.skip(reason="Requires actual model download - uncomment to test")
    def test_load_from_huggingface(self):
        """Test loading model from HuggingFace ID"""
        model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object")
        assert model.model is not None
        assert model.processor is not None
        assert model.device is not None
    
    @pytest.mark.skip(reason="Requires local model path - uncomment to test")
    def test_load_from_local_path(self):
        """Test loading model from local path"""
        model = OpenVLAOFTModel("/path/to/local/model")
        assert model.model is not None
        assert model.processor is not None
    
    @pytest.mark.skip(reason="Requires actual model and image - uncomment to test")
    def test_inference_with_real_model(self, sample_image):
        """Test running inference with actual model"""
        model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object")
        outputs = model.forward("Pick up the red block", sample_image)
        assert outputs is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
