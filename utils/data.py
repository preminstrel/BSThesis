import os


def count_file(path):
    """
    Count the number of files in a directory.
    """
    dir_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return len(dir_list)
