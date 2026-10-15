"""Shared predictor-date sample definitions for stages 05 through 05_03."""

SAMPLE_PERIODS = {
    'full_sample': ('2012-11-01', '2020-08-31'),
    'late_sample': ('2019-01-01', '2020-08-31'),
}

# Compatibility for commands used before the samples were unified.
SAMPLE_ALIASES = {'baseline': 'full_sample', '2019_2020': 'late_sample'}
