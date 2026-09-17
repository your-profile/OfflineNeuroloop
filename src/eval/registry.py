"""Registry of offline ML evaluation tests."""

from . import event_related, focus_transfer, lopo, positive_control, power, within_subject

TESTS = {
    "within_subject": within_subject.run,
    "lopo": lopo.run,
    "focus_transfer": focus_transfer.run,
    "positive_control": positive_control.run,
    "power": power.run,
    "event_related": event_related.run,
}


def get_test(name: str):
    if name not in TESTS:
        raise KeyError(f"Unknown test '{name}'. Choose from: {list(TESTS)}")
    return TESTS[name]


def list_tests():
    return sorted(TESTS)
