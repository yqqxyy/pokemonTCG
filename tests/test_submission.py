from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest
import torch

from poketcg.submission import REQUIRED_ARCHIVE_FILES, build_submission


def _fake_official_dir(root: Path) -> Path:
    official = root / "official"
    cg = official / "cg"
    cg.mkdir(parents=True)
    (cg / "__init__.py").write_text("", encoding="utf-8")
    (cg / "api.py").write_text("", encoding="utf-8")
    (cg / "game.py").write_text("", encoding="utf-8")
    (cg / "libcg.so").write_bytes(b"fake")
    (official / "deck.csv").write_text("".join(f"{index}\n" for index in range(60)))
    return official


def test_build_submission_has_root_layout_and_slim_checkpoint(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "model_config": {"model_type": "test"},
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "history": ["must not be packaged"],
        },
        checkpoint,
    )
    output = tmp_path / "submission.tar.gz"

    result = build_submission(checkpoint, output, official_dir=official)

    assert result["archive"] == str(output)
    with tarfile.open(output, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        assert names >= REQUIRED_ARCHIVE_FILES
        assert not any(name.startswith("submission/") for name in names)
        model_file = archive.extractfile("model.pt")
        assert model_file is not None
        packaged = torch.load(model_file, map_location="cpu", weights_only=False)
        config_file = archive.extractfile("agent_config.json")
        assert config_file is not None
        agent_config = json.load(config_file)
    assert set(packaged) == {"model_config", "model_state_dict"}
    assert agent_config["mode"] == "policy"


def test_build_submission_records_mcts_runtime_config(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)
    output = tmp_path / "submission.tar.gz"

    result = build_submission(
        checkpoint,
        output,
        official_dir=official,
        mcts_simulations=16,
        mcts_determinizations=1,
    )

    assert result["agent_config"]["mode"] == "mcts"
    assert result["agent_config"]["mcts"]["simulations"] == 16
    with tarfile.open(output, "r:gz") as archive:
        config_file = archive.extractfile("agent_config.json")
        assert config_file is not None
        assert json.load(config_file) == result["agent_config"]


def test_build_submission_records_planner_mcts_runtime_config(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    result = build_submission(
        checkpoint,
        tmp_path / "submission.tar.gz",
        official_dir=official,
        tactical_planner=True,
        planner_threshold=0.85,
        planner_weight=2.5,
        mcts_simulations=8,
    )

    assert result["agent_config"]["mode"] == "planner-mcts"
    assert result["agent_config"]["planner"] == {
        "enabled": True,
        "threshold": 0.85,
        "weight": 2.5,
        "confidence_routing": True,
        "profile": "mega-lucario-ex",
    }


def test_build_submission_rejects_invalid_deck(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    (official / "deck.csv").write_text("1\n2\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="exactly 60"):
        build_submission(
            checkpoint,
            tmp_path / "submission.tar.gz",
            official_dir=official,
        )


def test_packaged_main_supports_kaggle_exec_without_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)
    output = tmp_path / "submission.tar.gz"
    build_submission(checkpoint, output, official_dir=official)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with tarfile.open(output, "r:gz") as archive:
        archive.extractall(bundle)

    monkeypatch.chdir(bundle)
    monkeypatch.setattr(sys, "path", list(sys.path))
    environment: dict = {}
    source = (bundle / "main.py").read_text(encoding="utf-8")

    exec(compile(source, "main.py", "exec"), environment)

    assert "__file__" not in environment
    assert environment["agent"]({"select": None}) == list(range(60))
