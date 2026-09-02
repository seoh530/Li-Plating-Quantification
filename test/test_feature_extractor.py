import numpy as np

from modules.ml.feature_extractor import FeatureExtractor


def test_enhanced_features_are_computed_on_crop_coordinates() -> None:
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[:, :, 0] = np.arange(12, dtype=np.uint8).reshape(3, 4) * 3
    image[:, :, 1] = 80
    image[:, :, 2] = 160
    image[:, :, 3] = 255

    extractor = FeatureExtractor(
        local_residual_sigma_px=1.0,
        mm_per_pixel=0.5,
        edge_band_mm=0.5,
    )
    batch = extractor.extract(image, feature_columns=FeatureExtractor.ENHANCED_FEATURE_COLUMNS)

    assert batch.features.shape == (12, len(FeatureExtractor.ENHANCED_FEATURE_COLUMNS))
    columns = {name: idx for idx, name in enumerate(batch.feature_columns)}

    point_features, valid = extractor.select_point_features(batch, [(0, 0), (2, 1)])

    assert valid.tolist() == [True, True]
    assert point_features[0, columns["x_norm"]] == 0.0
    assert point_features[0, columns["y_norm"]] == 0.0
    assert np.isclose(point_features[1, columns["x_norm"]], 2 / 3)
    assert np.isclose(point_features[1, columns["y_norm"]], 1 / 2)
    assert point_features[0, columns["distance_to_edge_px"]] == 0.0
    assert point_features[0, columns["edge_band_5mm"]] == 1.0
    assert np.isfinite(point_features[:, columns["l_res"]]).all()
    assert np.isfinite(point_features[:, columns["l_robust"]]).all()
