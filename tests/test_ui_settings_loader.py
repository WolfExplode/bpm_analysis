"""ui_settings_loader: ui_settings.json parsing, key migration, and CLI defaulting."""
import json

import pytest

import ui_settings_loader as usl
from audio_preprocessing import CHANNEL_MODE_ALL, CHANNEL_MODE_MIXED
from config import DEFAULT_OUTPUT_OPTIONS, DEFAULT_PARAMS


# --- migrate_ui_settings_keys ------------------------------------------------

def test_migrate_maps_legacy_verbose_console_logging():
    settings = {"verbose_console_logging": True}
    usl.migrate_ui_settings_keys(settings)
    assert settings["algorithm_console_logging"] is True


def test_migrate_does_not_overwrite_existing_new_key():
    settings = {"algorithm_console_logging": False, "verbose_console_logging": True}
    usl.migrate_ui_settings_keys(settings)
    assert settings["algorithm_console_logging"] is False


def test_migrate_maps_legacy_general_debug_logging():
    settings = {"general_debug_logging": True}
    usl.migrate_ui_settings_keys(settings)
    assert settings["general_console_logging"] is True


def test_migrate_maps_process_all_channels_to_channel_mode():
    settings = {"process_all_channels": True}
    usl.migrate_ui_settings_keys(settings)
    assert settings["channel_mode"] == CHANNEL_MODE_ALL


def test_migrate_no_legacy_keys_is_a_no_op():
    settings = {"some_other_key": 1}
    usl.migrate_ui_settings_keys(settings)
    assert settings == {"some_other_key": 1}


def test_migrate_false_process_all_channels_does_not_set_channel_mode():
    settings = {"process_all_channels": False}
    usl.migrate_ui_settings_keys(settings)
    assert "channel_mode" not in settings


# --- load_ui_settings_json ----------------------------------------------------

def test_load_ui_settings_json_missing_file_returns_none(tmp_path):
    assert usl.load_ui_settings_json(str(tmp_path / "nope.json")) is None


def test_load_ui_settings_json_valid_dict(tmp_path):
    p = tmp_path / "ui_settings.json"
    p.write_text(json.dumps({"cli_output_dir": "out"}), encoding="utf-8")
    result = usl.load_ui_settings_json(str(p))
    assert result == {"cli_output_dir": "out"}


def test_load_ui_settings_json_applies_migration(tmp_path):
    p = tmp_path / "ui_settings.json"
    p.write_text(json.dumps({"verbose_console_logging": True}), encoding="utf-8")
    result = usl.load_ui_settings_json(str(p))
    assert result["algorithm_console_logging"] is True


def test_load_ui_settings_json_non_dict_returns_none(tmp_path):
    p = tmp_path / "ui_settings.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert usl.load_ui_settings_json(str(p)) is None


def test_load_ui_settings_json_malformed_returns_none(tmp_path):
    p = tmp_path / "ui_settings.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert usl.load_ui_settings_json(str(p)) is None


# --- parse_starting_bpm_from_settings -----------------------------------------

def test_parse_starting_bpm_valid_string():
    assert usl.parse_starting_bpm_from_settings({"starting_bpm": "  72.5  "}) == 72.5


def test_parse_starting_bpm_missing_key_returns_none():
    assert usl.parse_starting_bpm_from_settings({}) is None


def test_parse_starting_bpm_empty_string_returns_none():
    assert usl.parse_starting_bpm_from_settings({"starting_bpm": "   "}) is None


def test_parse_starting_bpm_non_numeric_returns_none():
    assert usl.parse_starting_bpm_from_settings({"starting_bpm": "fast"}) is None


# --- batch_cli_defaults_from_ui_settings --------------------------------------

def test_batch_cli_defaults_none_settings_uses_all_fallbacks():
    defaults = usl.batch_cli_defaults_from_ui_settings(None)
    assert defaults["jobs"] == 1
    assert defaults["output_dir"] == "processed_files"
    assert defaults["global_bpm_hint"] is None
    assert defaults["channel_mode"] == CHANNEL_MODE_MIXED
    assert defaults["use_springer_algorithm"] == DEFAULT_PARAMS["use_springer_algorithm"]
    assert defaults["springer_model"] == DEFAULT_PARAMS["springer_model"]


def test_batch_cli_defaults_non_numeric_jobs_falls_back_to_one():
    defaults = usl.batch_cli_defaults_from_ui_settings({"cli_batch_jobs": "many"})
    assert defaults["jobs"] == 1


def test_batch_cli_defaults_clamps_jobs_below_one():
    defaults = usl.batch_cli_defaults_from_ui_settings({"cli_batch_jobs": 0})
    assert defaults["jobs"] == 1


def test_batch_cli_defaults_accepts_valid_jobs():
    defaults = usl.batch_cli_defaults_from_ui_settings({"cli_batch_jobs": 4})
    assert defaults["jobs"] == 4


def test_batch_cli_defaults_blank_output_dir_falls_back():
    defaults = usl.batch_cli_defaults_from_ui_settings({"cli_output_dir": "   "})
    assert defaults["output_dir"] == "processed_files"


def test_batch_cli_defaults_strips_output_dir_whitespace():
    defaults = usl.batch_cli_defaults_from_ui_settings({"cli_output_dir": "  out/dir  "})
    assert defaults["output_dir"] == "out/dir"


def test_batch_cli_defaults_maps_output_option_keys():
    defaults = usl.batch_cli_defaults_from_ui_settings({"output_csv": True, "output_png": False})
    assert defaults["output_options"]["csv"] is True
    assert defaults["output_options"]["png"] is False


def test_batch_cli_defaults_unset_output_option_keeps_default():
    defaults = usl.batch_cli_defaults_from_ui_settings({})
    assert defaults["output_options"]["csv"] == DEFAULT_OUTPUT_OPTIONS["csv"]


def test_batch_cli_defaults_propagates_global_bpm_hint():
    defaults = usl.batch_cli_defaults_from_ui_settings({"starting_bpm": "90"})
    assert defaults["global_bpm_hint"] == 90.0


def test_batch_cli_defaults_invalid_channel_mode_raises():
    with pytest.raises(ValueError):
        usl.batch_cli_defaults_from_ui_settings({"channel_mode": "surround"})
