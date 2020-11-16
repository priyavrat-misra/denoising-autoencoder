from collections import namedtuple
from itertools import product


class RunBuilder():
    @staticmethod
    def get_runs(params):
        """
        build sets of parameters that define the runs.

        Args:
            params (OrderedDict): OrderedDict having hyper-parameter values

        Returns:
            list: containing list of all runs
        """
        Run = namedtuple('run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
