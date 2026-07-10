"""
Canonical label constants, collapse mappings, and shared prediction-loading
utilities for the LLM-vs-PolitiFact fact-checking analysis.

Centralizing these here avoids duplicated definitions across analysis scripts
and eliminates the sys.path hacks previously used in the sim_agreement
sub-pipeline to import from sibling scripts.
"""

import glob
import os

import pandas as pd


# ============================================================================
# Label constants
# ============================================================================

NEI = "Not enough information"

# The six PolitiFact Truth-O-Meter veracity labels. PolitiFact also publishes
# Flip-O-Meter ratings ("Half flip", "Full flop", "No flip"), which rate
# position changes rather than factuality and are excluded from this study.
VERACITY_VERDICTS = [
    "True",
    "Mostly true",
    "Half true",
    "Mostly false",
    "False",
    "Pants on fire",
]

MULTI_CLASS_ORDER = VERACITY_VERDICTS + [NEI]

TERNARY_ORDER = ["True", "Mixed", "False", NEI]
TO_TERNARY = {
    "True": "True",
    "Mostly true": "True",
    "Half true": "Mixed",
    "Mostly false": "Mixed",
    "False": "False",
    "Pants on fire": "False",
    NEI: NEI,
}

BINARY_ORDER = ["True", "False", NEI]
TO_BINARY = {
    "True": "True",
    "Mostly true": "True",
    "Half true": "True",
    "Mostly false": "False",
    "False": "False",
    "Pants on fire": "False",
    NEI: NEI,
}

# Maps scenario name → (label_map | None, ordered_label_list).
# `label_map=None` means multi_class (no collapse).
SCENARIOS = {
    "multi_class": (None, MULTI_CLASS_ORDER),
    "ternary": (TO_TERNARY, TERNARY_ORDER),
    "binary": (TO_BINARY, BINARY_ORDER),
}

# Glob pattern for cleaned prediction parquets (relative to data_dir).
PREDICTIONS_GLOB = "cleaned__model_*.parquet"

# Minimum number of claims a model group must have to be included in the
# common-claim intersection. Groups below this threshold are treated as
# partially-completed runs and excluded before intersecting, so a single
# short file cannot collapse the common set for every other model.
MIN_PREDICTIONS_PER_GROUP = 5


# ============================================================================
# Prediction loading
# ============================================================================


def load_predictions(data_dir, min_group_size=MIN_PREDICTIONS_PER_GROUP):
    """
    Load all cleaned prediction parquets from `data_dir` and return a tidy
    DataFrame restricted to the intersection of factcheck_analysis_link values
    present in every model group.

    Parameters
    ----------
    data_dir : str or path-like
        Directory containing cleaned__model_*.parquet files
        (i.e. the `data/processed/fc_results/` directory of this repo).
    min_group_size : int, optional
        Minimum number of claims a model group must have to be included in
        the intersection. Groups below this threshold are treated as
        partially-completed runs and excluded. Defaults to
        MIN_PREDICTIONS_PER_GROUP (5).

    Returns
    -------
    pandas.DataFrame
        Columns: factcheck_analysis_link, politifact_verdict, model,
        new_label. Null predictions and sparse groups (< min_group_size
        claims) are dropped before computing the common-claim intersection.

    Raises
    ------
    ValueError
        If any politifact_verdict value is not one of the six Truth-O-Meter
        veracity labels in VERACITY_VERDICTS.
    """
    files = sorted(glob.glob(os.path.join(data_dir, PREDICTIONS_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matching '{PREDICTIONS_GLOB}' in {data_dir}")

    df = pd.concat(
        [
            pd.read_parquet(
                f,
                columns=[
                    "factcheck_analysis_link",
                    "politifact_verdict",
                    "model",
                    "new_label",
                ],
            )
            for f in files
        ],
        ignore_index=True,
    )

    # Ground-truth verdicts must be Truth-O-Meter veracity labels. Anything
    # else (e.g. Flip-O-Meter ratings) would map to NaN in the ternary/binary
    # collapse maps and silently corrupt metrics, so fail loudly instead.
    bad_verdicts = ~df["politifact_verdict"].isin(VERACITY_VERDICTS)
    if bad_verdicts.any():
        bad_counts = df.loc[bad_verdicts, "politifact_verdict"].value_counts(
            dropna=False
        )
        raise ValueError(
            "Found non-veracity politifact_verdict values in cleaned predictions "
            f"(expected one of {VERACITY_VERDICTS}):\n{bad_counts.to_string()}\n"
            "Regenerate the sample with code/misc/sample_claims.py and re-run "
            "the cleaning pipeline."
        )

    null_pred = df["new_label"].isna()
    if null_pred.any():
        print("Null predictions per model — dropped before scoring:")
        print(df[null_pred].groupby("model").size().to_string())
        df = df[~null_pred].reset_index(drop=True)

    group_links = {
        name: set(g["factcheck_analysis_link"]) for name, g in df.groupby("model")
    }

    # Drop sparse groups (e.g. partially-completed runs) before intersecting,
    # otherwise a single 1-row file collapses the common set for every model.
    sparse = {
        name: len(s) for name, s in group_links.items() if len(s) < min_group_size
    }
    if sparse:
        print(
            f"\nSkipping {len(sparse)} sparse model groups "
            f"with < {min_group_size} claims:"
        )
        for name, n in sparse.items():
            print(f"\t- {name}: {n}")
        for name in sparse:
            del group_links[name]
        # Exclude sparse models directly — clearer than the equivalent inclusion
        # filter on the already-pruned group_links dict.
        df = df[~df["model"].isin(sparse.keys())].reset_index(drop=True)

    if not group_links:
        raise ValueError(
            f"No model groups remain after dropping sparse groups "
            f"(min_group_size={min_group_size}). Lower min_group_size or check that "
            f"data/processed/fc_results/ contains complete runs."
        )

    common = set.intersection(*group_links.values())
    print(f"\nCommon claims across {len(group_links)} model groups: {len(common):,}")
    df = df[df["factcheck_analysis_link"].isin(common)].reset_index(drop=True)
    return df
