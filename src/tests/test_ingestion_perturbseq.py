import pandas as pd

from napistu.ingestion import perturbseq
from napistu.ingestion.constants import (
    HARMONIZOME_DEFS,
    PERTURBSEQ_DIRECTIONS,
    PERTURBSEQ_PERTURBATION_TYPES,
)


def test_assign_predicted_direction():
    """Test assign_predicted_direction function with various perturbation types and values."""
    df = pd.DataFrame(
        {
            HARMONIZOME_DEFS.PERTURBATION_TYPE: (
                [PERTURBSEQ_PERTURBATION_TYPES.OE] * 4
                + [PERTURBSEQ_PERTURBATION_TYPES.KD] * 4
            ),
            HARMONIZOME_DEFS.STANDARDIZED_VALUE: [
                1.5,
                0.5,
                -0.5,
                -1.5,
                1.5,
                0.5,
                -0.5,
                -1.5,
            ],
            HARMONIZOME_DEFS.THRESHOLDED_VALUE: [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        }
    )

    result = perturbseq.assign_predicted_direction(df)

    expected = [
        PERTURBSEQ_DIRECTIONS.STRONG_ACTIVATION,  # OE, value > threshold
        PERTURBSEQ_DIRECTIONS.WEAK_ACTIVATION,  # OE, 0 < value <= threshold
        PERTURBSEQ_DIRECTIONS.WEAK_REPRESSION,  # OE, -threshold <= value < 0
        PERTURBSEQ_DIRECTIONS.STRONG_REPRESSION,  # OE, value < -threshold
        PERTURBSEQ_DIRECTIONS.STRONG_REPRESSION,  # KD, value > threshold (flipped)
        PERTURBSEQ_DIRECTIONS.WEAK_REPRESSION,  # KD, 0 < value <= threshold (flipped)
        PERTURBSEQ_DIRECTIONS.WEAK_ACTIVATION,  # KD, -threshold <= value < 0 (flipped)
        PERTURBSEQ_DIRECTIONS.STRONG_ACTIVATION,  # KD, value < -threshold (flipped)
    ]

    assert list(result) == expected
