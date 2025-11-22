
"""
This script loads the OpenVLA-OFT model from a given huggingface id or path and prepares it for inference. It uses the AutoModelForVision2Seq class from the transformers library.

Classes: 
    LoadOpenVLAOFT: Loads the OpenVLA-OFT model from a given huggingface id or path and prepares it for inference.
    OpenVLAOFTModel: Wrapper class for the OpenVLA-OFT model that provides a clean interface for inference.
Functions:
    load_openvla_oft: Loads the OpenVLA-OFT model from a given huggingface id or path and prepares it for inference.
    forward: Forward pass through the model.
    __call__: Alias for forward method.
Tests:
    test_load_openvla_oft: Tests the load_openvla_oft function.
    test_openvla_oft_model: Tests the OpenVLAOFTModel class.
    test_forward: Tests the forward method of the OpenVLAOFTModel class.
    test_inference: Tests the inference of the OpenVLAOFTModel class.
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path
from typing import Union, Optional


def load_openvla_oft(
    model_path_or_id: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = False,
    low_cpu_mem_usage: bool = True,
):
    """
    Load OpenVLA-OFT model from HuggingFace ID or local path. The model_kwargs are the keyword arguments for the from_pretrained method of the AutoModelForVision2Seq class.
    
    Args:
        model_path_or_id: HuggingFace model ID (e.g., "moojink/openvla-7b-oft-finetuned-libero-object")
                          or local path to model directory
        device: Device to load model on (e.g., "cuda:0", "cpu"). If None, auto-detects.
        torch_dtype: Data type for model weights (e.g., torch.bfloat16, torch.float16). If None, uses float32.
        use_flash_attention: Whether to use flash attention 2 (requires flash-attn package)
        low_cpu_mem_usage: Whether to use low CPU memory usage during loading
    
    Returns:
        tuple: (model, processor) - The loaded model and processor
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Default to bfloat16 if CUDA available, else float32
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Check if path is local or HuggingFace ID
    is_local = Path(model_path_or_id).exists()
    
    # Prepare loading kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "trust_remote_code": True,
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load processor
    print(f"Loading processor from {'local path' if is_local else 'HuggingFace'}...")
    processor = AutoProcessor.from_pretrained(
        model_path_or_id,
        trust_remote_code=True
    )
    
    # Load model
    print(f"Loading model from {'local path' if is_local else 'HuggingFace'}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path_or_id,
        **model_kwargs
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully on {device} with dtype {torch_dtype}")
    
    return model, processor


class OpenVLAOFTModel:
    """
    Wrapper class for OpenVLA-OFT model that provides a clean interface for inference.
    """
    
    def __init__(
        self,
        model_path_or_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = False,
    ):
        """
        Initialize OpenVLA-OFT model.
        
        Args:
            model_path_or_id: HuggingFace model ID or local path
            device: Device to load model on
            torch_dtype: Data type for model weights
            use_flash_attention: Whether to use flash attention 2
        """
        self.model, self.processor = load_openvla_oft(
            model_path_or_id=model_path_or_id,
            device=device,
            torch_dtype=torch_dtype,
            use_flash_attention=use_flash_attention,
        )
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, prompt: str, images, return_tensors: str = "pt", **kwargs):
        """
        Forward pass through the model.
        
        Args:
            prompt: Text instruction/prompt
            images: Single image (PIL Image) or list of images
            return_tensors: Format to return tensors ("pt" for PyTorch)
            **kwargs: Additional arguments passed to processor
        
        Returns:
            Model outputs
        """
        # Process inputs
        inputs = self.processor(
            prompt,
            images=images if isinstance(images, list) else [images],
            return_tensors=return_tensors,
            **kwargs
        ).to(self.device, dtype=self.model.dtype)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs
    
    def __call__(self, prompt: str, images, **kwargs):
        """Alias for forward method."""
        return self.forward(prompt, images, **kwargs)


if __name__ == "__main__":
    print("OpenVLA-OFT model loading script ready!")
    print("Use OpenVLAOFTModel class to load and use the model.")
    print("See test_oft_loading.py for usage examples.")
