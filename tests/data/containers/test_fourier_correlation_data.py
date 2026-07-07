import numpy as np
import pandas as pd
import pytest

from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)


def test_fcd_empty_construction():
    fcd = FourierCorrelationData()
    assert fcd.correlation["correlation"] is None
    assert fcd.resolution["threshold"] is None


def test_fcd_construction_with_data():
    fcd = FourierCorrelationData(
        data={
            "correlation": np.array([0.9, 0.5, 0.1]),
            "frequency": np.array([0.1, 0.5, 1.0]),
            "points-x-bin": np.array([100, 50, 10]),
        }
    )
    assert fcd.correlation["correlation"][0] == pytest.approx(0.9)
    assert fcd.correlation["frequency"][2] == pytest.approx(1.0)


def test_fcd_setitem_getitem():
    fcd = FourierCorrelationData()
    fcd.correlation["correlation"] = np.array([1.0, 0.8])
    fcd.resolution["threshold"] = 0.143
    assert np.array_equal(fcd.correlation["correlation"], np.array([1.0, 0.8]))
    assert fcd.resolution["threshold"] == 0.143


def test_fcd_immutable_key_rejection():
    fcd = FourierCorrelationData()
    with pytest.raises(KeyError, match="not defined"):
        fcd.correlation["bad-key"] = 1.0


def test_fcd_construction_with_bad_key():
    with pytest.raises(ValueError, match="Unknown key"):
        FourierCorrelationData(data={"bad-key": 42})


def test_fcd_construction_rejects_non_dict():
    with pytest.raises(TypeError, match="dict or None"):
        FourierCorrelationData(data=[1, 2, 3])


def test_fcd_as_dataframe_basic():
    fcd = FourierCorrelationData(
        data={
            "correlation": np.array([0.9, 0.5]),
            "frequency": np.array([0.1, 0.5]),
            "points-x-bin": np.array([100, 50]),
        }
    )
    df = fcd.as_dataframe(include_results=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["Correlation", "Frequency", "nPoints"]
    assert df["Correlation"].iloc[0] == pytest.approx(0.9)
    assert df["Frequency"].iloc[1] == pytest.approx(0.5)
    assert df["nPoints"].iloc[0] == 100


def test_fcd_as_dataframe_with_results():
    fcd = FourierCorrelationData(
        data={
            "correlation": np.array([0.9]),
            "frequency": np.array([0.1]),
            "points-x-bin": np.array([100]),
            "threshold": 0.143,
            "resolution": 1.5,
            "resolution-point": (10, 0.7),
        }
    )
    df = fcd.as_dataframe(include_results=True)
    assert "Resolution" in df.columns
    assert "Threshold" in df.columns
    assert df["Correlation"].iloc[0] == pytest.approx(0.9)
    assert df["Resolution"].iloc[0] == pytest.approx(1.5)
    assert df["Resolution_X"].iloc[0] == pytest.approx(10.0)
    assert df["Threshold"].iloc[0] == (0.143,)


def test_fcd_collection_setitem_getitem():
    coll = FourierCorrelationDataCollection()
    fcd = FourierCorrelationData()
    coll[0] = fcd
    assert coll[0] is fcd


def test_fcd_collection_multiple_angles():
    coll = FourierCorrelationDataCollection()
    fcd0 = FourierCorrelationData()
    fcd45 = FourierCorrelationData()
    coll[0] = fcd0
    coll[45] = fcd45
    assert coll[0] is fcd0
    assert coll[45] is fcd45


def test_fcd_collection_len():
    coll = FourierCorrelationDataCollection()
    assert len(coll) == 0
    coll[0] = FourierCorrelationData()
    assert len(coll) == 1


def test_fcd_collection_nitems():
    coll = FourierCorrelationDataCollection()
    assert coll.nitems() == 0
    coll[0] = FourierCorrelationData()
    assert coll.nitems() == 1


def test_fcd_collection_clear():
    coll = FourierCorrelationDataCollection()
    coll[0] = FourierCorrelationData()
    coll[90] = FourierCorrelationData()
    coll.clear()
    assert len(coll) == 0


def test_fcd_collection_items():
    coll = FourierCorrelationDataCollection()
    fcd = FourierCorrelationData()
    coll[0] = fcd
    items = coll.items()
    assert len(items) == 1
    assert items[0][1] is fcd


def test_fcd_collection_iteration():
    """Verifies the iterator fix (was returning self, now returns independent iterator)."""
    coll = FourierCorrelationDataCollection()
    coll[0] = FourierCorrelationData()
    coll[90] = FourierCorrelationData()
    items = list(coll)
    assert len(items) == 2
    # second iteration should produce all items again
    items2 = list(coll)
    assert len(items2) == 2


def test_fcd_collection_rejects_non_int_key():
    coll = FourierCorrelationDataCollection()
    with pytest.raises(TypeError, match="integer"):
        coll["bad"] = FourierCorrelationData()


def test_fcd_collection_rejects_non_fcd_value():
    coll = FourierCorrelationDataCollection()
    with pytest.raises(TypeError, match="FourierCorrelationData"):
        coll[0] = "not a FCD"


def test_fcd_collection_as_dataframe():
    coll = FourierCorrelationDataCollection()
    fcd = FourierCorrelationData(
        data={
            "correlation": np.array([0.9]),
            "frequency": np.array([0.1]),
            "points-x-bin": np.array([100]),
        }
    )
    coll[0] = fcd
    coll[90] = fcd
    df = coll.as_dataframe()
    assert len(df) == 2
    assert set(df["Angle"].unique()) == {0, 90}
    assert df["Correlation"].iloc[0] == pytest.approx(0.9)
    assert df["Frequency"].iloc[0] == pytest.approx(0.1)
    assert df["nPoints"].iloc[0] == 100
