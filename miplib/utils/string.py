def common_start(sa, sb):
    """ returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())

def common_string(strings):
    """
    Find the longest common string.

    :param strings:
    :return:
    """
    prefix1 = strings[0]
    prefix2 = strings[1]

    if prefix1.find('/') != -1:
        prefix1 = prefix1.split('/')
        prefix1 = prefix1[len(prefix1) - 1]

    if prefix2.find('/') != -1:
        prefix2 = prefix2.split('/')
        prefix2 = prefix2[len(prefix2) - 1]

    strings = [prefix1, prefix2]
    prefix = prefix1

    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break

    return prefix
