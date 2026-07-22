from __future__ import annotations

import tarfile
from pathlib import Path

from poketcg.external_submission import build_external_submission


def test_build_external_submission_uses_root_layout(tmp_path: Path) -> None:
    official = tmp_path / "official"
    cg = official / "cg"
    cg.mkdir(parents=True)
    for name in ("api.py", "game.py", "sim.py", "libcg.so"):
        (cg / name).write_bytes(b"binary" if name.endswith(".so") else b"# wrapper\n")
    (official / "deck.csv").write_text(
        "".join(f"{index}\n" for index in range(60))
    )
    source = tmp_path / "agent.py"
    source.write_text("def agent(obs, configuration=None):\n    return []\n")
    deck = tmp_path / "deck.csv"
    deck.write_text("".join(f"{index}\n" for index in range(60)))
    output = tmp_path / "submission.tar.gz"

    result = build_external_submission(
        source, deck, output, official_dir=official
    )

    assert result["deck_length"] == 60
    with tarfile.open(output, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        assert {"main.py", "deck.csv", "cg/api.py", "cg/libcg.so"} <= names
        assert archive.extractfile("main.py").read() == source.read_bytes()
