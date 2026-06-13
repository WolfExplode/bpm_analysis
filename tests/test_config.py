"""Config sanity + validate_params behavior."""
import logging

import config


def test_default_params_is_dict_of_known_keys():
    assert isinstance(config.DEFAULT_PARAMS, dict)
    assert config.DEFAULT_PARAMS  # non-empty
    # All keys are strings (used as kwargs / dict lookups throughout).
    assert all(isinstance(k, str) for k in config.DEFAULT_PARAMS)


def test_default_output_options_are_booleans():
    assert isinstance(config.DEFAULT_OUTPUT_OPTIONS, dict)
    assert all(isinstance(v, bool) for v in config.DEFAULT_OUTPUT_OPTIONS.values())


def test_bpm_bounds_are_ordered():
    p = config.DEFAULT_PARAMS
    assert p["min_bpm"] < p["max_bpm"]
    assert p["contractility_bpm_min"] < p["contractility_bpm_max"]


def test_validate_params_warns_once_for_unknown_key(caplog):
    # Reset the module-level dedupe set so this test is order-independent.
    config._warned_unknown_param_keys.clear()
    params = dict(config.DEFAULT_PARAMS)
    params["this_key_does_not_exist"] = 123
    with caplog.at_level(logging.WARNING):
        config.validate_params(params)
        config.validate_params(params)  # second call must NOT warn again
    warnings = [r for r in caplog.records if "this_key_does_not_exist" in r.getMessage()]
    assert len(warnings) == 1
    config._warned_unknown_param_keys.clear()


def test_validate_params_silent_for_known_keys(caplog):
    config._warned_unknown_param_keys.clear()
    with caplog.at_level(logging.WARNING):
        config.validate_params(dict(config.DEFAULT_PARAMS))
    assert not [r for r in caplog.records if "not in DEFAULT_PARAMS" in r.getMessage()]
