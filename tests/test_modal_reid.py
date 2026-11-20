from pathlib import Path

import pytest
import yaml

from tools.modal_reid_plan import export_argv, load_modal_config, train_argv, train_budget_minutes, volume_path

CFG = Path(__file__).resolve().parents[1] / "config" / "modal_reid.yaml"


def test_repo_config_loads_and_leaves_training_time():
    cfg = load_modal_config(CFG)
    assert 0 < train_budget_minutes(cfg) < cfg["timeout_minutes"]


@pytest.mark.parametrize("change", [{"extra": 1}, {"schema": "other"}, {"stop_margin_minutes": 120}])
def test_config_fails_closed(tmp_path, change):
    cfg = {**yaml.safe_load(CFG.read_text()), **change}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit):
        load_modal_config(p)


@pytest.mark.parametrize("bad", ["", "/etc/passwd", "../x", "runs/../../x"])
def test_volume_path_refuses_escape(bad):
    with pytest.raises(ValueError):
        volume_path("/vol", bad)


def test_train_argv_is_the_mac_cli_on_cuda():
    argv = train_argv("/vol", ["crops/a", "crops/b"], "backbones/d", "runs/r", "/root/config/reid_train.yaml", 105)
    assert argv[:4] == ["python", "-m", "tools.reid_train", "train"]
    assert argv[argv.index("--crops") + 1:argv.index("--backbone")] == ["/vol/crops/a", "/vol/crops/b"]
    assert argv[argv.index("--device") + 1] == "cuda" and argv[argv.index("--max-minutes") + 1] == "105"
    with pytest.raises(ValueError):
        train_argv("/vol", [], "backbones/d", "runs/r", "c.yaml", 1)


def test_export_argv_stays_in_volume():
    assert export_argv("/vol", "runs/r", "backbones/d", "models/m.onnx")[-1] == "/vol/models/m.onnx"
    with pytest.raises(ValueError):
        export_argv("/vol", "runs/r", "backbones/d", "../m.onnx")
