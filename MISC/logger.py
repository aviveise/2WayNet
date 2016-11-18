import datetime
import os

from singleton import Singleton

verbosity_map = {
    'info': 1,
    'debug': 2,
}


class OutputLog(object):
    __metaclass__ = Singleton

    def __init__(self):
        """
        Logging and output management
        """
        self.output_file_name = 'double_encoder'
        self._verbosity = verbosity_map['info']

    def write(self, message, verbosity='info'):
        if verbosity_map[verbosity] <= self._verbosity:
            print message
            self.output_file.write(message + '\n')
            self.output_file.flush()

    def set_path(self, path):
        self.path = path
        self.set_output_path(os.path.join(self.path, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

    def set_output_path(self, path, suffix=''):
        self.output_path = path
        if suffix:
            suffix += '_'
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.output_file = open(os.path.join(self.output_path, self.output_file_name + suffix + '.txt'), 'w+')

    def __del__(self):
        self.output_file.close()

    def set_verbosity(self, verbosity):
        self._verbosity = verbosity_map[verbosity]
