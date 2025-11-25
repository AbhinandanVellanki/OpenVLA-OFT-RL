"""
OpenVLA-OFT model loading and utilities.
"""

# Use absolute import to avoid pytest import issues
import sys
from pathlib import Path

# Add current directory to path if not already there
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from load_vla import OpenVLAOFTModel, load_openvla_oft
except ImportError:
    # Fallback to relative import if absolute fails
    from .load_vla import OpenVLAOFTModel, load_openvla_oft

__all__ = ['OpenVLAOFTModel', 'load_openvla_oft']
