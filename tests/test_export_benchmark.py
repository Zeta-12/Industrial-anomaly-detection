from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from industrial_ad import benchmark, viz

ov = pytest.importorskip("openvino")
from industrial_ad import export  # noqa: E402

Out = namedtuple("Out", "pred_score anomaly_map")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class TinyPatchModel(nn.Module):
    """Small PatchCore-like model: conv features + nearest neighbour in a random bank."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.register_buffer("bank", torch.randn(16, 8))

    def forward(self, x):
        f = self.conv(x)
        b, c, h, w = f.shape
        e = f.permute(0, 2, 3, 1).reshape(-1, c)
        d = e.pow(2).sum(-1, keepdim=True) + self.bank.pow(2).sum(-1)[None] - 2 * e @ self.bank.T
        patch = d.clamp_min(0).sqrt().min(dim=1).values.reshape(b, 1, h, w)
        return Out(pred_score=patch.amax(dim=(1, 2, 3)), anomaly_map=patch)


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return [rng.random((3, 32, 32), dtype=np.float32) for _ in range(4)]


def test_wrapper_applies_normalisation_then_model(images):
    model = TinyPatchModel().eval()
    wrapper = export.ExportWrapper(model, MEAN, STD).eval()
    x = torch.from_numpy(images[0])[None]
    score, amap = wrapper(x)
    expected = model((x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1))
    torch.testing.assert_close(score, expected.pred_score)
    assert amap.shape == (1, 32, 32)


def test_normalization_is_read_from_pre_processor():
    from torchvision.transforms import v2

    module = SimpleNamespace(
        pre_processor=SimpleNamespace(transform=v2.Compose([v2.Resize(32), v2.Normalize(MEAN, STD)]))
    )
    assert export.normalization(module) == (MEAN, STD)
    with pytest.raises(ValueError):
        export.normalization(SimpleNamespace(pre_processor=SimpleNamespace(transform=v2.Compose([v2.Resize(32)]))))


def test_openvino_fp32_matches_pytorch(images):
    wrapper = export.ExportWrapper(TinyPatchModel(), MEAN, STD).eval()
    compiled = ov.Core().compile_model(export.to_openvino(wrapper, (32, 32)), "CPU")
    scores, maps = benchmark.run_compiled(compiled, images)
    with torch.no_grad():
        ref = [wrapper(torch.from_numpy(x)[None]) for x in images]
    np.testing.assert_allclose(scores, [float(s) for s, _ in ref], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(maps[0], ref[0][1][0].numpy(), rtol=1e-4, atol=1e-5)


def test_int8_quantization_keeps_outputs_finite_and_close(images):
    pytest.importorskip("nncf")
    wrapper = export.ExportWrapper(TinyPatchModel(), MEAN, STD).eval()
    fp32 = export.to_openvino(wrapper, (32, 32))
    int8 = export.quantize_int8(fp32.clone(), images)
    core = ov.Core()
    s32, _ = benchmark.run_compiled(core.compile_model(fp32, "CPU"), images)
    s8, m8 = benchmark.run_compiled(core.compile_model(int8, "CPU"), images)
    assert np.isfinite(s8).all() and all(np.isfinite(m).all() for m in m8)
    np.testing.assert_allclose(s8, s32, rtol=0.1)


@pytest.mark.parametrize(
    ("target", "device", "config"),
    [
        ("CPU", "CPU", {"PERFORMANCE_HINT": "LATENCY"}),
        ("gpu:f32", "GPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}),
    ],
)
def test_parse_target(target, device, config):
    assert benchmark.parse_target(target) == (device, config)


def test_latency_stats_and_measurement():
    stats = benchmark.latency_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["latency_p50_ms"] == 2.5
    assert stats["fps_batch1"] == pytest.approx(400.0)
    calls = []
    measured = benchmark.measure_latency(lambda: calls.append(1), warmup=2, iterations=5)
    assert len(calls) == 7 and measured["iterations"] == 5
    with pytest.raises(ValueError):
        benchmark.latency_stats([])


ROWS = [
    {
        "runtime": "PyTorch",
        "target": "CPU",
        "weights": "fp32",
        "execution_precision": "f32",
        "latency_p50_ms": 400.0,
        "latency_p90_ms": 420.0,
        "fps_batch1": 2.5,
        "image_auroc": 0.949,
        "aupro": 0.940,
    },
    {
        "runtime": "OpenVINO",
        "target": "CPU",
        "weights": "int8",
        "execution_precision": "f32",
        "compile_s": 1.2,
        "latency_p50_ms": 120.0,
        "latency_p90_ms": 130.0,
        "fps_batch1": 8.3,
        "image_auroc": 0.945,
        "aupro": 0.938,
    },
    {
        "runtime": "OpenVINO",
        "target": "GPU",
        "weights": "fp16",
        "execution_precision": "f16",
        "compile_s": 20.0,
        "latency_p50_ms": 60.0,
        "latency_p90_ms": 70.0,
        "fps_batch1": 16.0,
        "accuracy_error": "3 non-finite outputs",
    },
    {"runtime": "OpenVINO", "target": "NPU", "weights": "fp32", "error": "RuntimeError: unsupported op TopK"},
]


def test_markdown_table_shows_failed_rows():
    table = benchmark.markdown_table(ROWS)
    lines = table.splitlines()
    assert len(lines) == 2 + len(ROWS)
    assert "**failed**: RuntimeError: unsupported op TopK" in table
    assert "**3 non-finite outputs**" in table
    assert "| 120.0 |" in table


def test_latency_accuracy_figure_skips_failed_rows(tmp_path):
    viz.save_latency_accuracy(tmp_path / "f.png", ROWS, "toy")
    assert (tmp_path / "f.png").stat().st_size > 0
