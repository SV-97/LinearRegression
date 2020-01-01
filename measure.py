from statistics import mean
import re

"""Regex to capture different components
Examples:
  0.000103 seconds (818 allocations: 87.328 KiB)
  0.000388 seconds (3.78 k allocations: 412.406 KiB)
  0.598357 seconds (2.02 M allocations: 102.555 MiB, 3.13% gc time)
"""
RE = r"\s*(?P<time>\d+\.\d*) seconds \((?:(?P<allocs>\d+(?:\.\d*)?)"\
    r"(?: (?P<allocsuffix>M|k))?) allocations: (?:(?P<mem>\d+\.\d*)"\
    r"(?: (?P<memsuffix>(?:KiB|MiB)))?)(?:, (?P<gctime>\d+\.\d*)% gc time)?\)"

MEMORY_FACTOR = {None: 1, "MiB": 1, "KiB": 1 / 1024}


def get_and_parse():
    try:
        while line := input():
            match = re.match(RE, line)
            time = float(match["time"])
            mem = float(match["mem"]) * MEMORY_FACTOR[match["memsuffix"]]
            if (gc_time:=match["gctime"]) is not None:
                gc = float(gc_time)
            else:
                gc = None
            yield (time, mem, gc)
    except EOFError: return None


time, memory, gc_time = zip(*(x for x in get_and_parse()))
gc_time = list(filter(lambda x: x is not None, gc_time))
print(
    f"Time: {mean(time):.5f} s, RAM: {mean(memory):.3f} MiB, "\
    f"GC-time: {mean(gc_time):.2f} % (got {len(gc_time)} gc values)")
