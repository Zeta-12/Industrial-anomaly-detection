import pytest

from industrial_ad import cli


@pytest.mark.parametrize("command", ["download", "train", "evaluate", "run", "summary", "export", "benchmark"])
def test_every_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0
    assert "usage: iad" in capsys.readouterr().out


def test_devices_runs_without_config(capsys):
    pytest.importorskip("openvino")
    cli.main(["devices"])
    out = capsys.readouterr().out
    assert "OpenVINO" in out and "CPU" in out
