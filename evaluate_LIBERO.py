"""
Evaluation script for OpenVLA on LIBERO benchmark tasks.

Evaluates a pretrained OpenVLA model on all tasks in a LIBERO task suite.
Supports both L1 regression head and tokenized action prediction.

Usage:
    # Evaluate on all tasks in libero_spatial (default: 20 episodes per task)
    python evaluate_LIBERO.py --task-suite libero_spatial
    
    # Evaluate specific tasks with custom episodes
    python evaluate_LIBERO.py --task-suite libero_spatial --task-ids 0 1 2 --num-episodes 50
    
    # Evaluate with tokenized actions (instead of L1 head)
    python evaluate_LIBERO.py --task-suite libero_10 --use-tokenized-actions
    
    # Evaluate with custom checkpoint
    python evaluate_LIBERO.py --task-suite libero_spatial --checkpoint /path/to/checkpoint
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add paths for imports
from vla_oft.min_vla.config import OpenVLAActorConfig
from vla_oft.min_vla.actor import OpenVLAActor
from vla_oft.min_vla.action_tokenizer import ActionTokenizer
from libero_rl.envs.libero_env import LiberoEnv
from libero_rl.utils.obs_utils import process_observation_for_vla
from libero_rl.utils.action_utils import process_action_for_libero
from libero_rl.utils.task_utils import (
    get_task,
    get_num_tasks,
    get_all_task_names,
    get_max_episode_length,
    TASK_SUITES,
)


class LiberoEvaluator:
    """
    Evaluator for OpenVLA on LIBERO benchmark tasks.
    
    Supports:
    - All LIBERO task suites (spatial, object, goal, 10, 90)
    - L1 regression head or tokenized action prediction
    - Multi-camera input (2 cameras: agentview + wrist)
    - Deterministic evaluation with greedy action selection
    """
    
    def __init__(
        self,
        vla_config: OpenVLAActorConfig,
        task_suite: str,
        num_episodes: int = 20,
        image_size: int = 224,
        eval_temperature: float = 0.0,
        seed: int = 0,
        use_tokenized_actions: bool = False,
        device: str = "cuda:0",
    ):
        """
        Initialize evaluator.
        
        Args:
            vla_config: OpenVLA actor configuration
            task_suite: LIBERO task suite name
            num_episodes: Number of evaluation episodes per task
            image_size: Input image size for VLA
            eval_temperature: Temperature for action sampling (0.0 = greedy)
            seed: Random seed
            use_tokenized_actions: If True, use tokenized actions; else use L1 head
            device: Device for evaluation
        """
        self.task_suite = task_suite
        self.num_episodes = num_episodes
        self.image_size = image_size
        self.eval_temperature = eval_temperature
        self.seed = seed
        self.use_tokenized_actions = use_tokenized_actions
        self.device = device
        
        # Validate task suite
        if task_suite not in TASK_SUITES:
            raise ValueError(
                f"Unknown task suite: {task_suite}. "
                f"Available: {TASK_SUITES}"
            )
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize VLA actor
        print(f"\n{'='*70}")
        print(f"Initializing OpenVLA for evaluation...")
        print(f"{'='*70}")
        self.vla_config = vla_config
        self.actor = OpenVLAActor(vla_config)
        
        # Set to eval mode
        self.actor.vla.eval()
        if self.actor.l1_action_head is not None:
            self.actor.l1_action_head.eval()
        
        # Initialize action tokenizer if using tokenized actions
        if use_tokenized_actions:
            print("Initializing action tokenizer...")
            self.action_tokenizer = ActionTokenizer(
                vocab_size=32000,
                n_bins=256,
                min_action=-1.0,
                max_action=1.0,
            )
            self.actor.vla.bin_centers = self.action_tokenizer.bin_centers
            self.actor.vla.vocab_size = self.action_tokenizer.vocab_size
            print(f"  {self.action_tokenizer}")
        else:
            self.action_tokenizer = None
        
        # Store unnorm_key for L1 head
        self.unnorm_key = "libero_spatial_no_noops"
        
        print(f"\n{'='*70}")
        print(f"Evaluation Configuration:")
        print(f"{'='*70}")
        print(f"  Task Suite: {task_suite}")
        print(f"  Num Tasks: {get_num_tasks(task_suite)}")
        print(f"  Episodes per task: {num_episodes}")
        print(f"  Action mode: {'Tokenized' if use_tokenized_actions else 'L1 Regression'}")
        print(f"  Temperature: {eval_temperature}")
        print(f"  Image size: {image_size}")
        print(f"  Num cameras: 2 (agentview + wrist)")
        print(f"  Device: {device}")
        print(f"  Seed: {seed}")
        print(f"{'='*70}\n")
    
    def get_action(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
    ) -> np.ndarray:
        """
        Get action chunk from VLA policy.
        
        Args:
            obs: Observation dictionary with 'image' and 'proprio'
            task_prompt: Task description string
        
        Returns:
            actions: (8, 7) numpy array of actions
        """
        if self.use_tokenized_actions:
            return self._get_action_via_tokens(obs, task_prompt)
        else:
            return self._get_action_via_predict(obs, task_prompt)
    
    def _get_action_via_predict(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
    ) -> np.ndarray:
        """
        Get action using VLA's built-in predict_action() method with L1 head.
        This matches the reference 98% evaluation implementation.
        """
        # Process observation for VLA
        prompt = f"In: What action should the robot take to {task_prompt.lower()}?\nOut:"
        
        # Get image (already PIL Image or list of PIL Images from processing)
        image = obs['image']
        
        # Handle multi-image input: concatenate pixel_values along image dimension
        if isinstance(image, list):
            # Process first image to get base inputs
            inputs = self.actor.processor(prompt, image[0]).to(
                self.device, dtype=torch.bfloat16
            )
            
            # Process additional images and concatenate pixel_values
            for additional_image in image[1:]:
                additional_inputs = self.actor.processor(prompt, additional_image).to(
                    self.device, dtype=torch.bfloat16
                )
                # Concatenate along the image dimension (dim=1)
                inputs["pixel_values"] = torch.cat(
                    [inputs["pixel_values"], additional_inputs["pixel_values"]], dim=1
                )
        else:
            # Single image
            inputs = self.actor.processor(prompt, image).to(
                self.device, dtype=torch.bfloat16
            )
        
        # Process proprioception if available
        proprio = None
        if self.vla_config.use_proprio and obs.get('proprio') is not None:
            proprio = obs['proprio']
            # Convert to numpy array if needed
            if not isinstance(proprio, np.ndarray):
                proprio = np.array(proprio, dtype=np.float32)
            # Normalize proprio using VLA's norm_stats
            if hasattr(self.actor.vla, 'norm_stats') and self.actor.vla.norm_stats is not None:
                proprio_mean = np.array(self.actor.vla.norm_stats[self.unnorm_key]["proprio"]["mean"], dtype=np.float32)
                proprio_std = np.array(self.actor.vla.norm_stats[self.unnorm_key]["proprio"]["std"], dtype=np.float32)
                proprio = (proprio - proprio_mean) / (proprio_std + 1e-8)
            proprio = torch.from_numpy(proprio).to(
                self.device, dtype=torch.bfloat16
            ).unsqueeze(0)
        
        # Call VLA's predict_action (with L1 action head if available)
        if self.actor.l1_action_head is not None:
            actions, _ = self.actor.vla.predict_action(
                **inputs,
                proprio=proprio,
                unnorm_key=self.unnorm_key,
                action_head=self.actor.l1_action_head,
            )
        else:
            raise ValueError(
                "L1 action head not loaded. Use --use-tokenized-actions or "
                "ensure load_l1_action_head=True in config."
            )
        
        # Convert to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        return actions
    
    def _get_action_via_tokens(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
    ) -> np.ndarray:
        """
        Get action using tokenized prediction (for models without L1 head).
        """
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX
        from ppo.core_algos import logprobs_from_logits
        
        # Prepare observation
        image = obs["image"]
        proprio = obs.get("proprio", None)
        
        # Clip proprio to expected dimension
        if proprio is not None and len(proprio) > 8:
            proprio = proprio[:8]
        
        # Build prompt
        prompt = f"In: What action should the robot take to {task_prompt.lower()}?\nOut:"
        
        # Convert single numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        
        # Handle multi-image input: concatenate pixel_values along image dimension
        if isinstance(image, list):
            # Process first image to get base inputs
            inputs = self.actor.processor(prompt, image[0]).to(
                self.device, dtype=torch.bfloat16
            )
            
            # Process additional images and concatenate pixel_values
            for additional_image in image[1:]:
                additional_inputs = self.actor.processor(prompt, additional_image).to(
                    self.device, dtype=torch.bfloat16
                )
                # Concatenate along the image dimension (dim=1)
                inputs["pixel_values"] = torch.cat(
                    [inputs["pixel_values"], additional_inputs["pixel_values"]], dim=1
                )
        else:
            # Single image
            inputs = self.actor.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]
        
        # Add empty token if needed (from predict_action)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        
        # Create fake labels
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        
        # Get number of prompt tokens
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
        
        # Prepare inputs (add action tokens)
        input_ids, attention_mask = self.actor.vla._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = self.actor.vla._prepare_labels_for_action_prediction(labels, input_ids)
        
        # Get input embeddings and action masks
        input_embeddings = self.actor.vla.get_input_embeddings()(input_ids)
        all_actions_mask = self.actor.vla._process_action_masks(labels)
        
        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        
        # Process vision features
        use_film = False
        projected_patch_embeddings = self.actor.vla._process_vision_features(pixel_values, language_embeddings, use_film)
        
        # Add proprio if available
        if self.actor.proprio_projector is not None and proprio is not None:
            proprio_tensor = torch.from_numpy(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self.actor.vla._process_proprio_features(
                projected_patch_embeddings, proprio_tensor, self.actor.proprio_projector
            )
        
        # Calculate number of patches
        NUM_PATCHES = self.actor.vla.vision_backbone.get_num_patches() * self.actor.vla.vision_backbone.get_num_images_in_input()
        if self.actor.proprio_projector is not None and proprio is not None:
            NUM_PATCHES += 1
        
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask
        
        # Build multimodal embeddings
        multimodal_embeddings, multimodal_attention_mask = self.actor.vla._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        
        # Forward through language model
        language_model_output = self.actor.vla.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract logits for action tokens
        action_logits = language_model_output.logits[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]
        
        # Extract last 256 tokens (action vocabulary)
        # FIXED: Should be [-256:] not [-256-64:-64] to get tokens 31744-32000
        action_token_logits = action_logits[..., -256:]
        
        # Greedy decoding (argmax)
        responses = torch.argmax(action_token_logits, dim=-1)
        
        # Detokenize to continuous action chunk (8 actions)
        responses_np = responses[0].detach().cpu().numpy()
        discretized_actions = self.action_tokenizer.vocab_size - responses_np
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.action_tokenizer.bin_centers.shape[0] - 1)
        continuous_actions = self.action_tokenizer.bin_centers[discretized_actions]
        continuous_actions = continuous_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        
        return continuous_actions
    
    def evaluate_task(
        self,
        task_id: int,
        task_order_index: int = 0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate on a single task.
        
        Args:
            task_id: Task index within suite
            task_order_index: Task ordering index
            verbose: If True, print episode-by-episode progress
        
        Returns:
            Dictionary with evaluation results
        """
        # Get task info
        task = get_task(self.task_suite, task_id, task_order_index)
        task_name = task.name
        task_language = task.language
        max_steps = get_max_episode_length(self.task_suite)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Task {task_id}: {task_name}")
            print(f"  Language: {task_language}")
            print(f"  Max steps: {max_steps}")
            print(f"{'='*70}")
        
        # Create environment
        env = LiberoEnv(
            task_suite_name=self.task_suite,
            task_id=task_id,
            task_order_index=task_order_index,
            seed=self.seed,
            num_steps_wait=10,
            render_mode=None,
            obs_mode="raw",  # Get full observation dict for processing
            action_normalization="none",  # Process actions manually
        )
        
        # Evaluation metrics
        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        
        # Run evaluation episodes
        with torch.inference_mode():
            for ep in tqdm(
                range(self.num_episodes),
                desc=f"Task {task_id}",
                disable=not verbose,
                ncols=100,
            ):
                obs, info = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                
                # Initialize action queue
                action_queue = deque(maxlen=8)
                
                while not done and step_count < max_steps:
                    # Query policy if action queue is empty
                    if len(action_queue) == 0:
                        # Process observation (2 cameras: agentview + wrist)
                        processed_obs = process_observation_for_vla(
                            obs,
                            camera_name="agentview",
                            resize_size=(self.image_size, self.image_size),
                            num_images=2,  # Two cameras required by model
                            center_crop=True,
                            crop_scale=0.9,
                            return_pil=True,
                        )
                        actor_obs = {
                            "image": processed_obs["image"],
                            "proprio": processed_obs["proprio"],
                        }
                        
                        # Get action chunk (8 actions)
                        actions_chunk = self.get_action(actor_obs, task_language)
                        
                        # Fill action queue
                        for act in actions_chunk:
                            action_queue.append(act)
                    
                    # Pop action from queue
                    action = action_queue.popleft()
                    
                    # Process action for LIBERO
                    action = process_action_for_libero(action)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(
                        action.tolist() if hasattr(action, 'tolist') else action
                    )
                    episode_reward += reward
                    done = terminated or truncated
                    step_count += 1
                
                # Record metrics
                episode_rewards.append(episode_reward)
                episode_successes.append(float(info.get("success", False)))
                episode_lengths.append(step_count)
                
                # Clear cache periodically
                if (ep + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Close environment
        env.close()
        
        # Compute statistics
        results = {
            "task_id": task_id,
            "task_name": task_name,
            "task_language": task_language,
            "num_episodes": self.num_episodes,
            "success_rate": np.mean(episode_successes),
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "all_successes": episode_successes,
            "all_rewards": episode_rewards,
            "all_lengths": episode_lengths,
        }
        
        if verbose:
            print(f"\nResults:")
            print(f"  Success Rate: {results['success_rate']*100:.1f}%")
            print(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
            print(f"  Mean Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
        
        return results
    
    def evaluate_suite(
        self,
        task_ids: Optional[List[int]] = None,
        task_order_index: int = 0,
        save_results: bool = True,
        output_dir: str = "./eval_results",
    ) -> Dict[str, Any]:
        """
        Evaluate on all tasks (or specified subset) in the task suite.
        
        Args:
            task_ids: List of task IDs to evaluate (None = all tasks)
            task_order_index: Task ordering index
            save_results: If True, save results to JSON file
            output_dir: Directory to save results
        
        Returns:
            Dictionary with aggregate results across all tasks
        """
        # Determine which tasks to evaluate
        num_tasks = get_num_tasks(self.task_suite, task_order_index)
        if task_ids is None:
            task_ids = list(range(num_tasks))
        
        print(f"\n{'='*70}")
        print(f"Evaluating {len(task_ids)} tasks from {self.task_suite}")
        print(f"{'='*70}")
        
        # Evaluate each task
        all_results = []
        for task_id in task_ids:
            task_results = self.evaluate_task(task_id, task_order_index, verbose=True)
            all_results.append(task_results)
            
            # Clear cache between tasks
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Aggregate results
        aggregate = {
            "task_suite": self.task_suite,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "num_episodes_per_task": self.num_episodes,
            "action_mode": "tokenized" if self.use_tokenized_actions else "l1_regression",
            "overall_success_rate": np.mean([r["success_rate"] for r in all_results]),
            "overall_mean_reward": np.mean([r["mean_reward"] for r in all_results]),
            "per_task_results": all_results,
            "eval_config": {
                "image_size": self.image_size,
                "eval_temperature": self.eval_temperature,
                "seed": self.seed,
                "num_cameras": 2,
            },
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Task Suite: {self.task_suite}")
        print(f"Num Tasks: {len(task_ids)}")
        print(f"Episodes per Task: {self.num_episodes}")
        print(f"Total Episodes: {len(task_ids) * self.num_episodes}")
        print(f"\nOverall Success Rate: {aggregate['overall_success_rate']*100:.2f}%")
        print(f"Overall Mean Reward: {aggregate['overall_mean_reward']:.3f}")
        print(f"\nPer-Task Success Rates:")
        for result in all_results:
            print(f"  Task {result['task_id']:2d} ({result['task_name'][:40]:40s}): {result['success_rate']*100:5.1f}%")
        print(f"{'='*70}\n")
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.task_suite}_eval_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(aggregate, f, indent=2)
            print(f"Results saved to: {filepath}\n")
        
        return aggregate


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate OpenVLA on LIBERO benchmark tasks"
    )
    
    # Task configuration
    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_spatial",
        choices=TASK_SUITES,
        help="LIBERO task suite to evaluate",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific task IDs to evaluate (default: all tasks)",
    )
    parser.add_argument(
        "--task-order-index",
        type=int,
        default=0,
        help="Task ordering index (0-19 for 10-task suites)",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="Temperature for action sampling (0.0 = greedy)",
    )
    
    # Model configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to custom checkpoint (default: use pretrained model)",
    )
    parser.add_argument(
        "--use-tokenized-actions",
        action="store_true",
        help="Use tokenized actions instead of L1 regression head",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for evaluation",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file",
    )
    
    args = parser.parse_args()
    
    # VLA configuration - defaults match reference run_libero_eval.py
    # Reference uses: use_l1_regression=True (equivalent to our load_l1_action_head=True)
    # But we allow --use-tokenized-actions flag for flexibility
    vla_config = OpenVLAActorConfig(
        load_l1_action_head=not args.use_tokenized_actions,  # True by default (matches reference)
        freeze_l1_action_head=True,
        use_tokenized_actions=args.use_tokenized_actions,  # False by default (matches reference)
        num_images_in_input=2,  # Required: model expects 2 cameras
        use_multi_gpu=False,  # Single GPU for evaluation
        gpu_id=0,  # Use GPU 0 for single-GPU evaluation
    )
    
    # Override checkpoint if provided
    if args.checkpoint is not None:
        vla_config.pretrained_checkpoint = args.checkpoint
    
    # Initialize evaluator
    evaluator = LiberoEvaluator(
        vla_config=vla_config,
        task_suite=args.task_suite,
        num_episodes=args.num_episodes,
        image_size=args.image_size,
        eval_temperature=args.eval_temperature,
        seed=args.seed,
        use_tokenized_actions=args.use_tokenized_actions,
        device=args.device,
    )
    
    # Run evaluation
    results = evaluator.evaluate_suite(
        task_ids=args.task_ids,
        task_order_index=args.task_order_index,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )
    
    return results


if __name__ == "__main__":
    main()
