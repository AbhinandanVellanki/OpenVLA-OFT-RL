"""
Minimal test suite for OpenVLA-OFT model loading.
"""

import pytest
from unittest.mock import Mock, patch

from load_vla import OpenVLAOFTModel


class TestOpenVLAOFTModel:
    """Basic test for OpenVLAOFTModel initialization."""
    
    def test_init_loads_model_and_processor(self):
        """Test that __init__ loads model and processor."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('load_vla.load_openvla_oft', return_value=(mock_model, mock_processor)):
            with patch('torch.cuda.is_available', return_value=False):
                model_wrapper = OpenVLAOFTModel("test-model-id")
                assert model_wrapper.model == mock_model
                assert model_wrapper.processor == mock_processor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
