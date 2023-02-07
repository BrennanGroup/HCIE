import os
import shutil
from functools import wraps
from tempfile import mkdtemp


def work_in_tmp_dir():
    """
    Executes a function in a temporary directory, which is then deleted after the function is complete
    :return: None
    """
    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            tmpdir_path = mkdtemp(dir=None)

            os.chdir(tmpdir_path)
            result = func(*args, **kwargs)

            os.chdir(here)
            shutil.rmtree(tmpdir_path)

            return result
        return wrapped_function
    return func_decorator
