import os
import sys
import datetime


class Logger(object):


    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = file_path

    def write(self, msg):

        self.terminal.write(msg)
        with open(self.log, 'a') as f:
            f.write(msg)

    def flush(self):
        pass


def record_log(file_path):

    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H：%M：%S")
    file_path = file_path + '/log_{}.log'.format(time_now)
    sys.stdout = Logger(file_path)