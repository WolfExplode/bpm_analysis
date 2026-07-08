"""console_logging: unicode-safe stream reconfiguration and Kaleido/choreographer noise filter."""
import logging

import console_logging as cl


def _record(name="root", pathname="", levelno=logging.INFO):
    return logging.LogRecord(
        name=name,
        level=levelno,
        pathname=pathname,
        lineno=1,
        msg="msg",
        args=None,
        exc_info=None,
    )


# --- SuppressKaleidoChoreographerRootNoiseFilter -----------------------------

def test_filter_passes_non_root_loggers():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(name="choreographer", levelno=logging.DEBUG)
    assert f.filter(rec) is True


def test_filter_passes_unrelated_root_records():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(name="root", pathname="/some/project/module.py", levelno=logging.DEBUG)
    assert f.filter(rec) is True


def test_filter_suppresses_choreographer_which_debug():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(
        name="root",
        pathname="/x/choreographer/utils/_which.py",
        levelno=logging.DEBUG,
    )
    assert f.filter(rec) is False


def test_filter_allows_choreographer_which_warning_and_above():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(
        name="root",
        pathname="/x/choreographer/utils/_which.py",
        levelno=logging.WARNING,
    )
    assert f.filter(rec) is True


def test_filter_handles_windows_style_backslash_paths():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(
        name="root",
        pathname="C:\\x\\choreographer\\utils\\_which.py",
        levelno=logging.DEBUG,
    )
    assert f.filter(rec) is False


def test_filter_suppresses_site_packages_kaleido_debug():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(
        name="root",
        pathname="/venv/lib/site-packages/kaleido/scopes/base.py",
        levelno=logging.DEBUG,
    )
    assert f.filter(rec) is False


def test_filter_allows_site_packages_kaleido_warning_and_above():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(
        name="root",
        pathname="/venv/lib/site-packages/kaleido/scopes/base.py",
        levelno=logging.WARNING,
    )
    assert f.filter(rec) is True


def test_filter_ignores_unrelated_site_packages():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(
        name="root",
        pathname="/venv/lib/site-packages/numpy/core.py",
        levelno=logging.DEBUG,
    )
    assert f.filter(rec) is True


def test_filter_handles_missing_pathname():
    f = cl.SuppressKaleidoChoreographerRootNoiseFilter()
    rec = _record(name="root", pathname="", levelno=logging.DEBUG)
    rec.pathname = None
    assert f.filter(rec) is True


# --- make_stream_unicode_safe -------------------------------------------------

class _FakeStream:
    def __init__(self, raises=False):
        self.raises = raises
        self.calls = []

    def reconfigure(self, **kwargs):
        if self.raises:
            raise RuntimeError("cannot reconfigure")
        self.calls.append(kwargs)


def test_make_stream_unicode_safe_calls_reconfigure():
    stream = _FakeStream()
    cl.make_stream_unicode_safe(stream)
    assert stream.calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]


def test_make_stream_unicode_safe_no_reconfigure_attr_does_not_raise():
    stream = object()  # no reconfigure attribute at all
    cl.make_stream_unicode_safe(stream)  # must not raise


def test_make_stream_unicode_safe_swallows_reconfigure_errors():
    stream = _FakeStream(raises=True)
    cl.make_stream_unicode_safe(stream)  # must not raise
