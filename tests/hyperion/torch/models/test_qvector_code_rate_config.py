"""Q-vector q-matrix code-rate configuration coverage."""

from jsonargparse import ArgumentParser

from hyperion.torch.models.qvectors.qvector import QVector


def test_qmatrix_code_rate_nested_parser_and_config():
    parser = ArgumentParser()
    QVector.add_class_args(parser)
    cfg = parser.parse_args(
        [
            "--enable-qmatrix-code-rate",
            "--qmatrix_code_rate.eps=0.25",
            "--qmatrix_code_rate.jitter=0.0002",
            "--qmatrix_code_rate.gamma-1=1.5",
            "--qmatrix_code_rate.normalize",
        ]
    ).as_dict()

    assert cfg["enable_qmatrix_code_rate"]
    assert cfg["qmatrix_code_rate"] == {
        "eps": 0.25,
        "jitter": 0.0002,
        "gamma_1": 1.5,
        "gamma_2": 1.0,
        "normalize": True,
    }
