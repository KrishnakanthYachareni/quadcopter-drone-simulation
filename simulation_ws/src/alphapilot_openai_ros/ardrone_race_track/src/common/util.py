"""Utility classes

Ref: https://github.com/blackredscarf/pytorch-DQN/blob/master/core/util.py
"""

__license__ = "Apache-2.0"
__author__ = "blackredscarf <blackredscarf@gmail.com>"

import os
import errno
import re
import time



def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('directory exists: % s' % parent_dir)
            else:
                print(e)
                raise

    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)

    if not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('directory exists: % s' % parent_dir)
            else:
                print(e)
                raise

    return parent_dir


def time_seq():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_class_attributes(cls):
    """Get attribute names from Class(type)
    :param cls: Class
    :return: List of class attributes
    """

    return [a for a, v in cls.__dict__.items()
            if not re.match('<function.*?>', str(v))
            and not (a.startswith('__') and a.endswith('__'))]


def get_class_attribute_values(cls):
    """Get class attribute names and their values from class(variable)
    :param cls: Class
    :return: A dictionary of class attribute names and their values
    """
    attr = get_class_attributes(type(cls))
    attr_dict = {}
    for a in attr:
        attr_dict[a] = getattr(cls, a)
    return attr_dict

def get_object_attribute_values(obj):
    """Get an instance object's attributes.

    :param obj: An object instance
    :return: A disctionary of object attribute names and their values.
    """
    return vars(obj)
