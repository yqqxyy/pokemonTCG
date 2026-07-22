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


def test_build_submission_records_plan_mcts_runtime_config(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    result = build_submission(
        checkpoint,
        tmp_path / "submission.tar.gz",
        official_dir=official,
        plan_mcts=True,
        plan_determinizations=3,
        plan_max_steps=24,
    )

    assert result["agent_config"]["mode"] == "plan-mcts"
    assert result["agent_config"]["planner"]["enabled"] is True
    assert result["agent_config"]["plan_mcts"] == {
        "enabled": True,
        "determinizations": 3,
        "max_macro_steps": 24,
        "root_contexts": [0],
        "prior": "fixed-model",
        "belief_hypotheses": [],
    }


def test_build_submission_embeds_plan_mcts_belief_decks(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)
    sample = tmp_path / "sample.csv"
    sample.write_text("".join(f"{1000 + index}\n" for index in range(60)))

    result = build_submission(
        checkpoint,
        tmp_path / "submission.tar.gz",
        official_dir=official,
        plan_mcts=True,
        plan_mcts_prior="belief",
        opponent_belief_decks=[("sample", sample)],
    )

    settings = result["agent_config"]["plan_mcts"]
    assert settings["prior"] == "belief"
    assert settings["belief_hypotheses"] == [
        {
            "name": "sample",
            "deck": list(range(1000, 1060)),
            "prior": 1.0,
        }
    ]


def test_build_submission_rejects_belief_without_hypotheses(
    tmp_path: Path,
) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="at least one"):
        build_submission(
            checkpoint,
            tmp_path / "submission.tar.gz",
            official_dir=official,
            plan_mcts=True,
            plan_mcts_prior="belief",
        )


def test_build_submission_rejects_duplicate_belief_names(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="Duplicate"):
        build_submission(
            checkpoint,
            tmp_path / "submission.tar.gz",
            official_dir=official,
            plan_mcts=True,
            plan_mcts_prior="belief",
            opponent_belief_decks=[
                ("same", official / "deck.csv"),
                ("same", official / "deck.csv"),
            ],
        )


def test_build_submission_rejects_plan_and_atomic_mcts(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="cannot be combined"):
        build_submission(
            checkpoint,
            tmp_path / "submission.tar.gz",
            official_dir=official,
            plan_mcts=True,
            mcts_simulations=8,
        )


def test_build_submission_records_standalone_mega_expert(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    result = build_submission(
        checkpoint,
        tmp_path / "submission.tar.gz",
        official_dir=official,
        mega_expert=True,
    )

    assert result["agent_config"]["mode"] == "mega-expert"
    assert result["agent_config"]["mega_expert"] == {"enabled": True}


def test_build_submission_rejects_mega_expert_hybrid(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="standalone"):
        build_submission(
            checkpoint,
            tmp_path / "submission.tar.gz",
            official_dir=official,
            mega_expert=True,
            plan_mcts=True,
        )


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
        "turn_ownership": False,
        "commitment_ownership": False,
        "profile": "mega-lucario-ex",
    }


def test_build_submission_records_turn_ownership(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    result = build_submission(
        checkpoint,
        tmp_path / "submission.tar.gz",
        official_dir=official,
        planner_turn_ownership=True,
    )

    assert result["agent_config"]["mode"] == "planner-policy"
    assert result["agent_config"]["planner"]["enabled"] is True
    assert result["agent_config"]["planner"]["turn_ownership"] is True


def test_build_submission_records_commitment_ownership(tmp_path: Path) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    result = build_submission(
        checkpoint,
        tmp_path / "submission.tar.gz",
        official_dir=official,
        planner_commitment_ownership=True,
    )

    assert result["agent_config"]["mode"] == "planner-policy"
    assert result["agent_config"]["planner"]["enabled"] is True
    assert result["agent_config"]["planner"]["turn_ownership"] is False
    assert result["agent_config"]["planner"]["commitment_ownership"] is True


def test_build_submission_rejects_turn_ownership_with_atomic_mcts(
    tmp_path: Path,
) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="cannot be combined"):
        build_submission(
            checkpoint,
            tmp_path / "submission.tar.gz",
            official_dir=official,
            planner_turn_ownership=True,
            mcts_simulations=8,
        )


def test_build_submission_rejects_commitment_ownership_with_atomic_mcts(
    tmp_path: Path,
) -> None:
    official = _fake_official_dir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="cannot be combined"):
        build_submission(
            checkpoint,
            tmp_path / "submission.tar.gz",
            official_dir=official,
            planner_commitment_ownership=True,
            mcts_simulations=8,
        )


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
