import numpy as np
import pytest
import torch

pytest.importorskip("anomalib")
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

from industrial_ad import int8_bank  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


@pytest.fixture
def tiny_patchcore():
    # untrained resnet18 so nothing is downloaded, layer2 + layer3 = 128 + 256 channels
    torch.manual_seed(0)
    pm = PatchcoreModel(layers=["layer2", "layer3"], backbone="resnet18", pre_trained=False).eval()
    pm.memory_bank = torch.randn(64, 384)
    return pm


def test_split_pipeline_reproduces_patchcore_forward(tiny_patchcore):
    x = torch.rand(2, 3, 64, 64)
    normalised = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    with torch.no_grad():
        reference = tiny_patchcore(normalised)
        embedding = int8_bank.FeatureWrapper(tiny_patchcore, MEAN, STD).eval()(x)
    score, amap = int8_bank.nn_head(tiny_patchcore, embedding, (64, 64))
    torch.testing.assert_close(score, reference.pred_score)
    torch.testing.assert_close(amap, reference.anomaly_map)


def test_build_bank_selects_a_subset_of_rows():
    embeddings = torch.randn(200, 16)
    bank = int8_bank.build_bank(embeddings, sampling_ratio=0.1, seed=0)
    assert bank.shape == (20, 16)
    assert all((embeddings == row).all(dim=1).any() for row in bank)
    torch.testing.assert_close(bank, int8_bank.build_bank(embeddings, sampling_ratio=0.1, seed=0))


def test_stage1_openvino_matches_pytorch(tiny_patchcore):
    ov = pytest.importorskip("openvino")
    wrapper = int8_bank.FeatureWrapper(tiny_patchcore, MEAN, STD).eval()
    with torch.no_grad():
        model = ov.convert_model(wrapper, example_input=torch.zeros(1, 3, 64, 64), input=[(1, 3, 64, 64)])
    compiled = ov.Core().compile_model(model, "CPU")
    x = np.random.default_rng(0).random((3, 64, 64), dtype=np.float32)
    scores, maps = int8_bank.score_two_stage(compiled, tiny_patchcore, [x], (64, 64))
    with torch.no_grad():
        ref_score, ref_map = int8_bank.nn_head(tiny_patchcore, wrapper(torch.from_numpy(x)[None]), (64, 64))
    np.testing.assert_allclose(scores[0], float(ref_score[0]), rtol=1e-4)
    np.testing.assert_allclose(maps[0], ref_map.reshape(64, 64).numpy(), rtol=1e-3, atol=1e-4)


def test_markdown_table():
    row = {
        "variant": "INT8 features + INT8 bank",
        "image_threshold": 30.1,
        "image_auroc": 0.94,
        "aupro": 0.93,
        "recall": 0.9,
        "fpr": 0.1,
        "spearman_vs_pytorch": 0.97,
        "stage1_p50_ms": 50.0,
        "stage2_p50_ms": 150.0,
    }
    table = int8_bank.markdown_table([row])
    assert "| INT8 features + INT8 bank | 30.10 | 0.940 | 0.930 |" in table
