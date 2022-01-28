import os
import re
import sys
import time


class Logger(object):
    """Duplicates all stdout to a file."""

    def __init__(self, path, resume, iternum):
        if not resume and os.path.exists(path):
            print(path + " exists")
            sys.exit(0)

        if not (os.path.exists(path)):
            with open(path, "w") as f:
                f.write("init time{}\n".format(time.asctime(time.localtime(time.time()))))

        if resume:
            with open(path, "r") as f:
                for line in f.readlines():
                    match = re.search("Iteration (\d+).* ", line)
                    if match is not None:
                        it = int(match.group(1))
                        if it > iternum:
                            iternum = it
        self.iternum = iternum

        self.log = open(path, "a", encoding="utf-8")  # if resume else open(path, "w")
        self.stdout = sys.stdout
        # sys.stdout = self

    def write(self, message):
        # self.stdout.write(message)
        # self.stdout.flush()
        self.log.write("Iteration " + message)
        self.log.flush()

    def flush(self):
        pass
