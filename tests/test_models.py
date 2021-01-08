import seisbench.models


def test_weights_docstring():
    model = seisbench.models.GPD()

    assert model.weights_docstring is None
    model.load_pretrained("dummy")
    assert isinstance(model.weights_docstring, str)
