from conditional_node_field_graph_generator.scientific_observations import (
    ObservationPolicy,
    detect_observations,
)


def test_observation_detector_reports_gap_plateau_and_gradient_instability():
    records = []
    for epoch in range(1, 9):
        records.append(
            {
                "epoch": epoch,
                "metrics": {"train_total": 0.2, "val_total": 0.35},
                "gradient_norm": 10.0 if epoch < 8 else 100.0,
                "epoch_duration_seconds": 1.0,
                "finite_metrics": True,
            }
        )
    observations = detect_observations(
        records,
        policy=ObservationPolicy(
            plateau_window_epochs=8,
            plateau_minimum_improvement=0.01,
            generalisation_gap_threshold=0.1,
            gradient_norm_threshold=50.0,
        ),
    )
    types = {observation["type"] for observation in observations}
    assert {"validation_plateau", "generalisation_gap", "unstable_gradients"} <= types


def test_observation_detector_reports_non_finite_metrics():
    observations = detect_observations(
        [
            {
                "epoch": 1,
                "metrics": {},
                "gradient_norm": None,
                "epoch_duration_seconds": 1.0,
                "finite_metrics": False,
            }
        ]
    )
    assert observations[0]["type"] == "non_finite_metric"
