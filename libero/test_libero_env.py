import importlib
import sys
import types

import numpy as np
import pytest  # type: ignore


class FakeVectorEnv:
	"""Lightweight stand-in for the real SubprocVectorEnv used in tests."""

	def __init__(self, env_creators):
		self.envs = [creator() for creator in env_creators]
		self.num_envs = len(self.envs)

	def _obs_template(self):
		return {"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8)}

	def seed(self, seed):
		self.seed_value = seed

	def reset(self, id=None):
		count = self.num_envs if id is None else len(id)
		return [self._obs_template() for _ in range(count)]

	def set_init_state(self, init_state, id=None):
		count = len(init_state) if isinstance(init_state, list) else 1
		return [self._obs_template() for _ in range(count)]

	def step(self, action, id=None):
		count = len(id) if id is not None else len(action)
		obs = [self._obs_template() for _ in range(count)]
		rewards = [0.0] * count
		dones = [False] * count
		infos = [{} for _ in range(count)]
		return obs, rewards, dones, infos

	def close(self):
		pass


@pytest.fixture(autouse=True)
def install_dependency_stubs(monkeypatch):
	"""Provide lightweight stand-ins for heavy dependencies before import."""

	from dataclasses import dataclass
	from typing import Any

	experiments_pkg = types.ModuleType("experiments")
	experiments_pkg.__path__ = []
	monkeypatch.setitem(sys.modules, "experiments", experiments_pkg)

	experiments_robot_pkg = types.ModuleType("experiments.robot")
	experiments_robot_pkg.__path__ = []
	experiments_pkg.robot = experiments_robot_pkg
	monkeypatch.setitem(sys.modules, "experiments.robot", experiments_robot_pkg)

	robot_utils_module = types.ModuleType("experiments.robot.robot_utils")
	robot_utils_module.DATE_TIME = "20250101"
	robot_utils_module.get_image_resize_size = lambda cfg: 224
	robot_utils_module.invert_gripper_action = lambda action: action
	robot_utils_module.normalize_gripper_action = lambda action, binarize=True: np.asarray(action)
	experiments_robot_pkg.robot_utils = robot_utils_module
	monkeypatch.setitem(sys.modules, "experiments.robot.robot_utils", robot_utils_module)

	openvla_utils_module = types.ModuleType("experiments.robot.openvla_utils")
	openvla_utils_module.preprocess_input = lambda img, prompt, pre_thought=None, center_crop=False: (img, prompt)
	openvla_utils_module.preprocess_input_batch = (
		lambda imgs, prompts, pre_thought_list=None, center_crop=False: (imgs, prompts)
	)
	experiments_robot_pkg.openvla_utils = openvla_utils_module
	monkeypatch.setitem(sys.modules, "experiments.robot.openvla_utils", openvla_utils_module)

	libero_pkg = types.ModuleType("libero")
	libero_pkg.__path__ = []
	monkeypatch.setitem(sys.modules, "libero", libero_pkg)

	class DummyTask:
		language = "dummy task"
		problem_folder = "dummy_folder"
		bddl_file = "dummy.bddl"

	class DummySuite:
		n_tasks = 1

		def get_task(self, task_id):
			return DummyTask()

		def get_task_init_states(self, task_id):
			return [{"state": task_id}]

	benchmark_namespace = types.SimpleNamespace(
		get_benchmark_dict=lambda: {"dummy_suite": lambda: DummySuite()}
	)

	libero_libero_module = types.ModuleType("libero.libero")
	libero_libero_module.benchmark = benchmark_namespace
	libero_libero_module.get_libero_path = lambda name: f"/tmp/{name}"
	libero_pkg.libero = libero_libero_module
	monkeypatch.setitem(sys.modules, "libero.libero", libero_libero_module)

	class StubOffScreenRenderEnv:
		def __init__(self, **kwargs):
			self.kwargs = kwargs

		def seed(self, seed):
			self.seed = seed

		def reset(self, id=None):
			return None

		def set_init_state(self, init_state, id=None):
			count = len(init_state) if isinstance(init_state, list) else 1
			return [
				{"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8)} for _ in range(count)
			]

		def step(self, action, id=None):
			count = len(id) if id is not None else len(action)
			obs = [
				{"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8)} for _ in range(count)
			]
			return obs, [0.0] * count, [False] * count, [{} for _ in range(count)]

		def close(self):
			pass

	libero_envs_module = types.ModuleType("libero.libero.envs")
	libero_envs_module.OffScreenRenderEnv = StubOffScreenRenderEnv
	monkeypatch.setitem(sys.modules, "libero.libero.envs", libero_envs_module)

	ppo_pkg = types.ModuleType("ppo")
	ppo_pkg.__path__ = []
	monkeypatch.setitem(sys.modules, "ppo", ppo_pkg)

	ppo_envs_pkg = types.ModuleType("ppo.envs")
	ppo_envs_pkg.__path__ = []
	ppo_pkg.envs = ppo_envs_pkg
	monkeypatch.setitem(sys.modules, "ppo.envs", ppo_envs_pkg)

	ppo_envs_base_module = types.ModuleType("ppo.envs.base")

	@dataclass
	class EnvOutput:
		pixel_values: Any
		prompts: Any

	class BaseEnv:
		def __init__(self, seed=None):
			self.seed = seed

	ppo_envs_base_module.BaseEnv = BaseEnv
	ppo_envs_base_module.EnvOutput = EnvOutput
	ppo_envs_pkg.base = ppo_envs_base_module
	monkeypatch.setitem(sys.modules, "ppo.envs.base", ppo_envs_base_module)

	ppo_envs_venv_module = types.ModuleType("ppo.envs.venv")
	ppo_envs_venv_module.SubprocVectorEnv = FakeVectorEnv
	ppo_envs_pkg.venv = ppo_envs_venv_module
	monkeypatch.setitem(sys.modules, "ppo.envs.venv", ppo_envs_venv_module)

	ppo_utils_pkg = types.ModuleType("ppo.utils")
	ppo_utils_pkg.__path__ = []
	ppo_pkg.utils = ppo_utils_pkg
	monkeypatch.setitem(sys.modules, "ppo.utils", ppo_utils_pkg)

	ppo_utils_util_module = types.ModuleType("ppo.utils.util")
	ppo_utils_util_module.add_info_board = lambda img, **kwargs: img
	ppo_utils_pkg.util = ppo_utils_util_module
	monkeypatch.setitem(sys.modules, "ppo.utils.util", ppo_utils_util_module)

	tf_module = types.ModuleType("tensorflow")
	tf_module.image = types.SimpleNamespace(
		encode_jpeg=lambda img: img,
		resize=lambda img, size, method=None, antialias=None: img,
	)
	tf_module.io = types.SimpleNamespace(
		decode_image=lambda img, expand_animations=False, dtype=None: img
	)
	tf_module.cast = lambda tensor, dtype: tensor
	tf_module.clip_by_value = lambda tensor, low, high: tensor
	tf_module.round = lambda tensor: tensor
	tf_module.uint8 = np.uint8
	monkeypatch.setitem(sys.modules, "tensorflow", tf_module)


def test_vla_env_reset_and_step(monkeypatch):
	libero_env = importlib.reload(importlib.import_module("libero.libero_env"))

	monkeypatch.setattr(libero_env, "SubprocVectorEnv", FakeVectorEnv)
	monkeypatch.setattr(
		libero_env,
		"get_libero_image",
		lambda obs, resize_size: np.zeros((resize_size, resize_size, 3), dtype=np.uint8),
	)
	monkeypatch.setattr(
		libero_env,
		"preprocess_input_batch",
		lambda imgs, prompts, pre_thought_list=None, center_crop=False: (imgs, prompts),
	)

	cfg = types.SimpleNamespace(
		seed=0,
		env_gpu_id=0,
		num_tasks_per_suite=1,
		n_rollout_threads=1,
		task_ids=None,
		max_env_length=0,
		num_steps_wait=0,
		model_family="openvla",
		save_video=False,
		center_crop=False,
		exp_dir="/tmp",
		num_trials_per_task=1,
		penalty_reward_value=-1.0,
		non_stop_penalty=False,
		verify_reward_value=1.0,
		task_suite_name="dummy_suite",
	)

	env = libero_env.VLAEnv(cfg, mode="train")

	obs, info = env.reset()
	assert len(obs.pixel_values) == env.env_num
	assert len(obs.prompts) == env.env_num
	assert info["step_count"].shape == (env.env_num,)
	assert info["task_description"][0] == "dummy task"

	action = np.zeros((env.env_num, 7), dtype=np.float32)
	next_obs, rewards, dones, infos = env.step(action)

	assert rewards.shape == (env.env_num,)
	assert np.allclose(rewards, 0.0)
	assert dones.shape == (env.env_num,)
	assert not np.any(dones)
	assert len(next_obs.pixel_values) == env.env_num
	assert infos["step_count"][0] == 1

	env.close()
