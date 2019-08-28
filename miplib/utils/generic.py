def isiterable(something):
    """ Check if a variable is iterable
    
    :param something: some variable that you are interested in
    :type something: any
    :return: True/False
    :rtype: boolean
    """
    try:
        iter(something)
        return True
    except TypeError:
        return False