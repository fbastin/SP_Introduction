# LShaped.jl vs StochasticPrograms.jl: benchmark results

Results of `LShaped_benchmark.jl` (all instances, best of 3 runs), started with 6 threads
(`julia -t 6`), on 5 October 2026:

- Intel Core i5-8500 (6 cores), 23 GB of memory;
- Julia 1.12.5, JuMP 1.31.2, MathOptInterface 1.53.0, HiGHS.jl 1.25.1;
- StochasticPrograms.jl from https://github.com/fbastin/StochasticPrograms.jl at `362da81`, which
  does not use Julia's threads.

Times in seconds, the best of three runs, model building included; memory allocated by one run, in
MB. Every method converged, to within 10⁻⁶ of the extensive form in relative terms. The variants of
LShaped.jl: dropping the optimality cuts slack at 5 consecutive iterates (`drop_inactive = 5`),
and solving the recourse problems on the 6 threads (`threads = true`).

| Instance | Method | Iterations | Cuts | Time (s) | Alloc. (MB) |
|:--|:--|--:|--:|--:|--:|
| farmer, S = 100 | extensive form |  |  | 0.009 | 2 |
|  | LShaped.jl, single-cut | 19 | 18 | 0.145 | 9 |
|  | LShaped.jl, single-cut, drop 5 | 19 | 18 | 0.147 | 9 |
|  | LShaped.jl, multicut | 7 | 496 | 0.080 | 6 |
|  | LShaped.jl, multicut, 6 threads | 7 | 496 | 0.038 | 6 |
|  | StochasticPrograms, single-cut | 23 | 20 | 1.066 | 69 |
|  | StochasticPrograms, multicut | 9 | 496 | 0.678 | 49 |
| farmer, S = 1000 | extensive form |  |  | 0.192 | 20 |
|  | LShaped.jl, single-cut | 22 | 21 | 1.593 | 101 |
|  | LShaped.jl, single-cut, drop 5 | 22 | 21 | 1.587 | 101 |
|  | LShaped.jl, multicut | 8 | 4988 | 1.188 | 102 |
|  | LShaped.jl, multicut, 6 threads | 8 | 4988 | 0.645 | 102 |
|  | StochasticPrograms, single-cut | 24 | 22 | 11.018 | 699 |
|  | StochasticPrograms, multicut | 10 | 4988 | 7.448 | 525 |
| capacity, P = F = 10, S = 100 | extensive form |  |  | 0.110 | 31 |
|  | LShaped.jl, single-cut | 71 | 70 | 1.128 | 63 |
|  | LShaped.jl, single-cut, drop 5 | 66 | 65 | 1.077 | 62 |
|  | LShaped.jl, multicut | 17 | 1249 | 0.427 | 31 |
|  | LShaped.jl, multicut, 6 threads | 17 | 1249 | 0.174 | 31 |
|  | StochasticPrograms, single-cut | 69 | 67 | 6.130 | 1144 |
|  | StochasticPrograms, multicut | 17 | 1268 | 1.984 | 368 |
| capacity, P = F = 20, S = 500 | extensive form |  |  | 6.430 | 468 |
|  | LShaped.jl, single-cut | 159 | 158 | 25.974 | 1099 |
|  | LShaped.jl, single-cut, drop 5 | 188 | 187 | 30.435 | 1270 |
|  | LShaped.jl, multicut | 23 | 9789 | 8.157 | 400 |
|  | LShaped.jl, multicut, 6 threads | 23 | 9789 | 3.799 | 401 |
|  | StochasticPrograms, single-cut | 152 | 150 | 169.019 | 42340 |
|  | StochasticPrograms, multicut | 21 | 8582 | 31.347 | 7048 |
| no shortage, P = F = 10, S = 100 | extensive form |  |  | 0.095 | 25 |
|  | LShaped.jl, single-cut | 57 | 155 | 1.009 | 67 |
|  | LShaped.jl, single-cut, drop 5 | 57 | 155 | 1.031 | 69 |
|  | LShaped.jl, multicut | 17 | 1303 | 0.548 | 45 |
|  | LShaped.jl, multicut, 6 threads | 17 | 1303 | 0.200 | 45 |
|  | StochasticPrograms, single-cut | 55 | 152 | 4.840 | 850 |
|  | StochasticPrograms, multicut | 17 | 1258 | 1.828 | 330 |
| no shortage, P = F = 20, S = 200 | extensive form |  |  | 1.112 | 177 |
|  | LShaped.jl, single-cut | 131 | 329 | 8.979 | 459 |
|  | LShaped.jl, single-cut, drop 5 | 146 | 344 | 9.638 | 501 |
|  | LShaped.jl, multicut | 27 | 4578 | 3.833 | 251 |
|  | LShaped.jl, multicut, 6 threads | 27 | 4578 | 1.392 | 251 |
|  | StochasticPrograms, single-cut | 132 | 329 | 57.554 | 14073 |
|  | StochasticPrograms, multicut | 23 | 3686 | 12.546 | 2835 |

## Observations

- **Both codes solve every instance correctly**, with about the same number of iterations: the
  differences, of one to four, come from StochasticPrograms evaluating a starting point before its
  first master problem.
- **LShaped.jl is 4.8 to 7.3 times faster in its single-cut version, 3.3 to 8.5 times in its
  multicut version, on one thread**, and allocates 8 to 38 times less memory: 1.1 GB against 42 GB
  for the single-cut version on the largest instance.
- **Both multicut versions add only the violated cuts**, hence comparable cut counts. In LShaped.jl,
  this removed 10 to 30% of the cuts it added before (9 789 instead of 11 000 on the largest
  instance, 4 578 instead of 5 200 on the largest one without shortage).
- **Solving the recourse problems on 6 threads** makes the multicut version 2.1 to 2.8 times faster
  on the larger instances, with the same iterates and cuts. On the largest instance, it beats the
  extensive form solved directly by HiGHS: 3.8 s against 6.4 s.
- **Dropping the inactive cuts does not pay off here.** With a window of 5 iterations, the long
  single-cut runs take more iterations, because the dropped cuts must be generated again: 188
  instead of 159 on the largest instance, 146 instead of 131 on the largest one without shortage.
  It is off by default.
- **The code optimizations of LShaped.jl** (direct models for problems given as data, constraints
  read and written by typed groups, constant data converted once, the master objective rebuilt only
  when it changes) cut its times by 14 to 47% and its memory by 55 to 77% with respect to the
  previous version, the most on many small recourse problems: 2.34 s to 1.59 s and 163 MB to 101 MB
  for the single-cut version on the farmer with 1000 scenarios.
- **What remains is mostly the solver**: about 0.45 ms per recourse problem, of which 0.13 ms is a
  fixed cost per call that neither presolve, the dual simplex nor direct mode reduce.
- **On one thread, neither code beats the extensive form** solved directly by HiGHS: these are
  linear programs HiGHS solves at once, and decomposition pays off when the extensive form becomes
  too large, or, as above, when the recourse problems are solved in parallel.
