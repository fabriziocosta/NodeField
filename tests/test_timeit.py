import re
import time

import logging

from conditional_node_field_graph_generator.runtime_utils import _verbosity_level, timeit


class _Worker:
    def __init__(self, verbose):
        self.verbose = verbose

    @timeit
    def compute(self, x):
        time.sleep(0.001)
        return x + 1


def test_verbosity_level_handles_bool_and_int_and_missing():
    class _NoVerbose:
        pass

    assert _verbosity_level(_NoVerbose()) == 0
    assert _verbosity_level(_Worker(verbose=False)) == 0
    assert _verbosity_level(_Worker(verbose=True)) == 1
    assert _verbosity_level(_Worker(verbose=2)) == 2
    assert _verbosity_level(_Worker(verbose="3")) == 3


def test_timeit_logs_only_at_verbose_level_3(caplog):
    quiet = _Worker(verbose=2)
    with caplog.at_level(logging.INFO):
        assert quiet.compute(4) == 5
    assert caplog.text == ""

    loud = _Worker(verbose=3)
    with caplog.at_level(logging.INFO):
        assert loud.compute(9) == 10
    assert "Function 'compute' executed in" in caplog.text
    assert re.search(r"Class '_Worker', Function 'compute' executed in", caplog.text)
