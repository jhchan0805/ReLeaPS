import os
import sys
import cProfile
import multiprocessing
import numpy as np
from . import benchmark_all_pool, benchmark_circle_pool, benchmark_okabe_pool, benchmark_greedy_pool, \
  benchmark_random_pool, _all_algos, _algo_processes

def main():
  multiprocessing.set_start_method("spawn")
  for algo, processes in zip(_all_algos, _algo_processes):
    with multiprocessing.Pool(processes=processes) as pool:
      if sys.argv[1] == "all":
        benchmark_all_pool(pool, algo, sys.argv[2])
      elif sys.argv[1] == "circle":
        benchmark_circle_pool(pool, algo, sys.argv[2], int(sys.argv[3]), sys.argv[4].lower() == "true")
      elif sys.argv[1] == "okabe":
        rng = np.random.default_rng(0)
        benchmark_okabe_pool(pool, algo, sys.argv[2], int(sys.argv[3]), rng)
      elif sys.argv[1] == "greedy":
        benchmark_greedy_pool(pool, algo, sys.argv[2], int(sys.argv[3]))
      elif sys.argv[1] == "random":
        rng = np.random.default_rng(0)
        benchmark_random_pool(pool, algo, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), rng)
      else:
        print("Unknown operation:", sys.argv[1])
        sys.exit(-1)

if __name__ == "__main__":
  # cProfile.run("main()")
  main()
