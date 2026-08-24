"""Optional Temporal dependency boundary."""


class TemporalDependencyMissing(RuntimeError):
    """The optional temporalio dependency is unavailable."""


__all__ = ["TemporalDependencyMissing"]
