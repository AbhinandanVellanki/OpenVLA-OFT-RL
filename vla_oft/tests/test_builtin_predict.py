"""Quick test to verify VLA's built-in predict_action works"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, 'vla_oft')

from min_vla.config import OpenVLAActorConfig
from min_vla.actor import OpenVLAActor
from PIL import Image

print("="*70)
print("Testing VLA's Built-in predict_action()")
print("="*70)

# Create config
cfg = OpenVLAActorConfig(
    pretrained_checkpoint="openvla-7b-oft-finetuned-libero-spatial",
    device="cuda:1",
    use_proprio=True,
    load_l1_action_head=True,
)

# Load actor
print("\nLoading VLA actor...")
actor = OpenVLAActor(cfg)

# Check norm_stats
print(f"\nVLA norm_stats loaded: {hasattr(actor.vla, 'norm_stats') and actor.vla.norm_stats is not None}")
if hasattr(actor.vla, 'norm_stats') and actor.vla.norm_stats:
    print(f"  Keys: {list(actor.vla.norm_stats.keys())}")
    stats = actor.vla.norm_stats['libero_spatial_no_noops']
    print(f"  Action stats: {list(stats['action'].keys())}")
    print(f"  Action mask: {stats['action']['mask']}")

# Create dummy observation
print("\nCreating dummy observation...")
dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
dummy_proprio = np.random.randn(8).astype(np.float32)

# Build prompt
prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"

# Process inputs
print("\nProcessing inputs...")
inputs = actor.processor(prompt, dummy_img).to('cuda:1', dtype=torch.bfloat16)
print(f"  input_ids shape: {inputs['input_ids'].shape}")
print(f"  pixel_values shape: {inputs['pixel_values'].shape}")

# Test predict_action with L1 head
print("\nCalling vla.predict_action() with L1 action head...")
with torch.no_grad():
    actions, _ = actor.vla.predict_action(
        **inputs,
        unnorm_key="libero_spatial_no_noops",
        do_sample=False,
        proprio=dummy_proprio,
        proprio_projector=actor.proprio_projector,
        action_head=actor.l1_action_head,
        use_film=False,
    )

print(f"\n✓ Success!")
print(f"  Actions shape: {actions.shape}")
print(f"  Actions type: {type(actions)}")
print(f"  First action: {actions[0]}")
print(f"  Action ranges:")
for i in range(7):
    print(f"    Dim {i}: [{actions[:, i].min():.3f}, {actions[:, i].max():.3f}]")

print("\n" + "="*70)
print("Test completed successfully!")
print("="*70)
