"""Diagnostic controls must not change scientific budgets or admit CPU fits."""
import copy

import pytest
import pandas as pd

from icl_pilot.evaluation.framework_classifier import update_budget, cpu_diagnostic_backend, _write_classifier_dataset
from icl_pilot.evaluation.framework_data import load_protocol
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scientific_update_budget_is_unchanged():
    settings = load_protocol(ROOT)['classifier']
    assert update_budget(198, settings) == 75
    assert 'test_only' not in settings and 'diagnostic_max_steps' not in settings


@pytest.mark.parametrize('test_only,steps', [(False, 1), (None, 1), (True, 0), (True, 3), (True, True)])
def test_short_budget_requires_explicit_bounded_diagnostic(test_only, steps):
    settings = copy.deepcopy(load_protocol(ROOT)['classifier'])
    settings.update(test_only=test_only, diagnostic_max_steps=steps)
    with pytest.raises(ValueError, match='diagnostic update budget'):
        update_budget(198, settings)


@pytest.mark.parametrize('test_only,inner,steps', [(False, True, 1), (True, False, 1), (True, True, 3)])
def test_cpu_smoke_cannot_be_used_for_scientific_or_long_training(test_only, inner, steps):
    protocol = {'test_only': test_only, 'classifier': {'test_only': inner}}
    with pytest.raises(ValueError):
        cpu_diagnostic_backend({'max_steps': steps}, None, None, None, protocol, None)


def test_different_source_metadata_cannot_break_the_toolkit_csv_schema(tmp_path):
    synthetic = pd.DataFrame({'output_text': ['her go .', 'he goes .'], 'Label': [0, 1],
        'bootstrap_draw_id': ['a', 'b'], 'model_key': ['qwen3_8b'] * 2})
    real = pd.DataFrame({'output_text': ['him goed .', 'he went .'], 'Label': [0, 1],
        'child_key': ['enni:1', 'enni:2'], 'heldout_fold': ['fold_02'] * 2})
    before = [synthetic.copy(deep=True), real.copy(deep=True)]
    _write_classifier_dataset(tmp_path, synthetic, real)
    train, validation = [pd.read_csv(tmp_path / (split + '.csv')) for split in ['train', 'validation']]
    assert train.columns.tolist() == validation.columns.tolist() == ['output_text', 'Label']
    assert train['output_text'].tolist() == synthetic['output_text'].tolist()
    assert validation['Label'].tolist() == real['Label'].tolist()
    pd.testing.assert_frame_equal(synthetic, before[0])
    pd.testing.assert_frame_equal(real, before[1])
