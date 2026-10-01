"""engine: pure helpers (auto-switch-algorithm decision logic)."""
import engine


def test_should_switch_when_alternate_passes():
    assert engine._should_switch_algorithm(alt_failed=False, primary_reason_count=2, alt_reason_count=0) is True


def test_should_switch_when_alternate_fails_less():
    assert engine._should_switch_algorithm(alt_failed=True, primary_reason_count=3, alt_reason_count=1) is True


def test_should_not_switch_on_tie():
    assert engine._should_switch_algorithm(alt_failed=True, primary_reason_count=2, alt_reason_count=2) is False


def test_should_not_switch_when_alternate_fails_more():
    assert engine._should_switch_algorithm(alt_failed=True, primary_reason_count=1, alt_reason_count=3) is False
