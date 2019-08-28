"""
Various utilities that are used to convert command line parameters into
data types that the progrma understands.
"""
import os

file_extensions = ['.tif', '.lsm', 'tiff', '.raw', '.data']


def get_user_input(message):
    """
    A method to ask question. The answer needs to be yes or no.

    Parameters
    ----------
    :param message  string, the question

    Returns
    -------

    Return a boolean: True for Yes, False for No
    """
    while True:
        answer = input(message)
        if answer in ('y', 'Y', 'yes', 'YES'):
            return True
        elif answer in ('n', 'N', 'no', 'No'):
            return False
        else:
            print("Unkown command. Please state yes or no")


def get_path_dir(path, suffix):
    """ Return a directory name with suffix that will be used to save data
    related to given path.
    """
    if os.path.isfile(path):
        path_dir = path + '.' + suffix
    elif os.path.isdir(path):
        path_dir = os.path.join(path, suffix)
    elif os.path.exists(path):
        raise ValueError('Not a file or directory: %r' % path)
    else:
        base, ext = os.path.splitext(path)
        if ext in file_extensions:
            path_dir = path + '.' + suffix
        else:
            path_dir = os.path.join(path, suffix)
    return path_dir


def get_full_path(path, prefix):
    """
    :param path:    Path to a file (string)
    :param prefix:  Path prefix, if applicable. Used in cases in
                    which the path argument is not an absolute
                    path
    :return:        Returns the absolute path, if the file is found,
                    None otherwise
    """
    if not os.path.isfile(path):
        path = os.path.join(prefix, path)
        if not os.path.isfile(path):
            raise ValueError('Not a valid file %s' % path)
    return path


def get_filename_and_extension(path):
    """
    Returns a filename and the file extension. The filename cna
    be either a simlpe filename or a full path.

    :param path:
    :return:
    """
    filename = path.split('.')[:-1]
    extension = path.split('.')[-1]

    return filename, extension
