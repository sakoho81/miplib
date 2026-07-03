from collections.abc import Sequence


def common_start(sa: str, sb: str) -> str:
    """Return the longest common prefix of two strings."""

    def _iter():
        for a, b in zip(sa, sb, strict=False):
            if a == b:
                yield a
            else:
                return

    return "".join(_iter())


def common_string(strings: Sequence[str]) -> str:
    """Return the longest common prefix of the basenames of all strings."""
    basenames = []
    for s in strings:
        basenames.append(s.rsplit("/", 1)[-1])

    if not basenames:
        return ""

    prefix = basenames[0]
    for s in basenames[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
