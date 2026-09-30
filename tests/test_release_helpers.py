import pytest
import torch

from locowm.scripts import runtime
from locowm.scripts.runtime import PROJECT_ROOT, existing_file, output_path, project_path
from loco_rl.modules.adapter import Adapter
from loco_rl.modules.world_model import WorldModel


def test_project_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert project_path("checkpoints/model.pt") == PROJECT_ROOT / "checkpoints/model.pt"


def test_existing_file_resolves_repository_relative_paths(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    checkpoint = project_root / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setattr(runtime, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tmp_path)
    assert existing_file("checkpoints/model.pt", label="checkpoint") == str(checkpoint)


def test_existing_file_reports_resolved_path_for_missing_file():
    with pytest.raises(FileNotFoundError, match="Resolved path"):
        existing_file("missing/checkpoint.pt", label="checkpoint")


def test_output_path_defaults_under_repository():
    assert output_path(None, "logs", "eval", "task").startswith(str(PROJECT_ROOT))


def test_world_model_masks_targets_after_episode_boundary():
    model = WorldModel({"chunk_len": 3, "num_targets": 2, "mlp_hidden_dims": [4]})
    model.build(num_obs=2, num_actions=1, num_envs=1, num_transitions_per_env=4)
    model.targets[:, 0] = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    targets, mask = model._build_chunks(torch.tensor([[[0.0]], [[1.0]], [[0.0]], [[0.0]]]))

    assert targets.shape == (4, 1, 3, 2)
    assert mask[:, 0, :, 0].tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def test_adapter_residual_starts_at_zero():
    adapter = Adapter(3, 2, 4, 2, {"hidden_dims": [5, 4]})
    output = adapter(torch.randn(6, 3), torch.randn(6, 2), torch.randn(6, 4))
    assert torch.equal(output, torch.zeros_like(output))
