# Stochastic Programming — Course Materials

This repository contains the lecture slides, background materials, reference notes, and code scripts for a university-level course on **Stochastic Programming**.

---

## 📁 Repository Structure

The project is structured into dedicated, organized subdirectories:

```
SP_Introduction/
├── README.md                      # Course overview & directory guide
├── Makefile                       # Automated build script for all lectures & background notes
├── .gitignore                     # Git rules ignoring LaTeX build artifacts
├── slides/                        # Beamer presentation source files (.tex)
│   ├── 01. Introduction.tex
│   ├── 02. Random Numbers.tex
│   ├── 03. Two-stage stochastic programming.tex
│   ├── 04. L-Shaped method.tex
│   ├── 05. Chance-constrained programming.tex
│   ├── 06. Multistage.tex
│   ├── 07. SDDP.tex
│   ├── 08. Lagrangian.tex
│   ├── 09. SAA.tex
│   ├── 10. Adaptive Monte Carlo.tex
│   ├── 11. Stochastic gradient descent.tex
│   └── 12. Discrete event simulation.tex
├── pdf/                           # Compiled PDF slides and background notes
│   ├── 01. Introduction.pdf
│   └── ...
├── background/                    # Background notes, built by `make background`
│   ├── probability_background.tex # Probability & statistics prerequisites
│   ├── lp_background.tex          # Linear programming fundamentals
│   └── kkt_background.tex         # KKT optimality conditions
├── exercises/                     # Exercise collection; the source builds two PDFs
│   ├── exercises.tex              # 18 exercises, statements + solutions
│   └── Exercices/                 # Superseded French originals (PSTricks, not built)
├── sty/                           # Custom LaTeX macro definitions and style packages
│   ├── macros.tex                 # Common mathematical vector & matrix shortcuts
│   ├── crayola.sty                # Crayola color palette definitions
│   ├── eclbkbox.sty               # Breakable box environment style
│   └── mathlist.sty               # Custom list environment for math slides
├── imgs/                          # Figure, diagram, and plot assets (EPS, PNG, PDF, JPG)
└── code/                          # Associated notebooks and scripts
    ├── random.ipynb                          # Generators, LFSR, ziggurat (deck 02)
    ├── lfsr.jl, lfsr_test.jl                 # LFSR in Julia, with its tests (deck 02)
    ├── Farmer.ipynb                          # The farmer problem (decks 01, 03)
    ├── newsvendor.ipynb                      # The newsvendor problem (decks 01, 03)
    ├── two_stages.ipynb                      # Two-stage models (deck 03)
    ├── lshaped_general.ipynb                 # The L-shaped method (deck 04)
    ├── chance constrained toy example.ipynb  # Chance-constrained LP (deck 05)
    ├── portfolio-chanceconstrainedprogramming.ipynb  # Chance-constrained portfolio (deck 05)
    ├── portfolio_stochastic_programming.ipynb # Portfolio as a stochastic program
    ├── SDDP_from_scratch.ipynb               # SDDP step by step (deck 07)
    ├── SDDP_hydro.ipynb                      # Hydro-thermal examples with SDDP.jl (deck 07)
    ├── simulation_callcenter.ipynb           # Call-centre simulation (deck 12)
    ├── simulation_portfolio.ipynb            # Portfolio simulation (decks 10, 12)
    ├── StochasticPrograms/                   # Julia project: Farmer and L-shaped examples
    └── cp.plot                               # Gnuplot script
```

---

## 📚 Course Topics & Outline

| Topic # | Title | TeX Source | Compiled PDF | Key Concepts |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Introduction** | [`slides/01. Introduction.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/01.%20Introduction.tex) | [`pdf/01. Introduction.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/01.%20Introduction.pdf) | Motivation, stochasticity in optimization, recourse vs wait-and-see |
| **02** | **Random Numbers** | [`slides/02. Random Numbers.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/02.%20Random%20Numbers.tex) | [`pdf/02. Random Numbers.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/02.%20Random%20Numbers.pdf) | Pseudo-random number generation, LCGs, inversion & rejection sampling |
| **03** | **Two-Stage Stochastic Programming** | [`slides/03. Two-stage stochastic programming.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/03.%20Two-stage%20stochastic%20programming.tex) | [`pdf/03. Two-stage stochastic programming.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/03.%20Two-stage%20stochastic%20programming.pdf) | First vs second stage decisions, recourse functions, EVPI & VSS |
| **04** | **L-Shaped Method** | [`slides/04. L-Shaped method.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/04.%20L-Shaped%20method.tex) | [`pdf/04. L-Shaped method.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/04.%20L-Shaped%20method.pdf) | Benders decomposition for SP, optimality and feasibility cuts |
| **05** | **Chance-Constrained Programming** | [`slides/05. Chance-constrained programming.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/05.%20Chance-constrained%20programming.tex) | [`pdf/05. Chance-constrained programming.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/05.%20Chance-constrained%20programming.pdf) | Probabilistic constraints, quantile optimization, convex approximations |
| **06** | **Multistage Stochastic Programming** | [`slides/06. Multistage.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/06.%20Multistage.tex) | [`pdf/06. Multistage.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/06.%20Multistage.pdf) | Scenario trees, non-anticipativity constraints, dynamic formulations |
| **07** | **SDDP** | [`slides/07. SDDP.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/07.%20SDDP.tex) | [`pdf/07. SDDP.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/07.%20SDDP.pdf) | Stochastic Dual Dynamic Programming, forward/backward passes, cuts |
| **08** | **Lagrangian Relaxation** | [`slides/08. Lagrangian.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/08.%20Lagrangian.tex) | [`pdf/08. Lagrangian.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/08.%20Lagrangian.pdf) | Dual decomposition, non-anticipativity multiplier updates |
| **09** | **Sample Average Approximation (SAA)** | [`slides/09. SAA.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/09.%20SAA.tex) | [`pdf/09. SAA.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/09.%20SAA.pdf) | Monte Carlo sampling, statistical convergence, candidate selection |
| **10** | **Adaptive Monte Carlo** | [`slides/10. Adaptive Monte Carlo.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/10.%20Adaptive%20Monte%20Carlo.tex) | [`pdf/10. Adaptive Monte Carlo.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/10.%20Adaptive%20Monte%20Carlo.pdf) | Variance reduction techniques, importance sampling, CRN |
| **11** | **Stochastic Gradient Descent** | [`slides/11. Stochastic gradient descent.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/11.%20Stochastic%20gradient%20descent.tex) | [`pdf/11. Stochastic gradient descent.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/11.%20Stochastic%20gradient%20descent.pdf) | Stochastic approximation, Robbins-Monro, subgradient methods |
| **12** | **Discrete Event Simulation** | [`slides/12. Discrete event simulation.tex`](file:///home/bastin/slash/Git/SP_Introduction/slides/12.%20Discrete%20event%20simulation.tex) | [`pdf/12. Discrete event simulation.pdf`](file:///home/bastin/slash/Git/SP_Introduction/pdf/12.%20Discrete%20event%20simulation.pdf) | Event-driven simulation, queueing models, gradient estimation |

### Background & Prerequisites

Three self-contained decks, built by `make background`. Students missing the
prerequisites should read `probability_background` first, then `lp_background`;
`kkt_background` is only needed from deck 05 onwards.

| Deck | Source | Covers | Needed from |
| :--- | :--- | :--- | :--- |
| Probability | [`background/probability_background.tex`](file:///home/bastin/slash/Git/SP_Introduction/background/probability_background.tex) | Probability space, support, quantiles, moments, Jensen, conditional expectation, LLN/CLT, confidence intervals | deck 01 |
| Linear programming | [`background/lp_background.tex`](file:///home/bastin/slash/Git/SP_Introduction/background/lp_background.tex) | Standard form, bases and vertices, duality, Farkas, the value function and its subgradients | deck 03 |
| KKT | [`background/kkt_background.tex`](file:///home/bastin/slash/Git/SP_Introduction/background/kkt_background.tex) | Lagrangian, duality gap, KKT conditions, constraint qualifications | deck 05 |

### Exercise collection

`exercises/exercises.tex` holds the whole collection, organized by topic:

| Section | Exercises | Depends on |
| :--- | :---: | :--- |
| Generalities | 1 | deck 03 |
| Two-stage problems with recourse | 8 | decks 01, 03 |
| Decomposition methods | 2 | deck 04 |
| Multistage problems | 1 | decks 06, 08 |
| Random number generation | 5 | deck 02 |
| Monte Carlo approximation | 1 | decks 03, 09 |

It is a **single source producing two PDFs**, so statements never have to be
kept in sync with their solutions:

| PDF | Contents | Built with |
| :--- | :--- | :--- |
| `pdf/exercises.pdf` | statements only — the version to hand out | `\def\nosolutions{}` on the command line |
| `pdf/exercises_solutions.pdf` | statements **and** solutions | the default |

Solutions live in a `solution` environment, which the `comment` package removes
wholesale in the student version. Every figure is TikZ: the collection builds
with a plain `pdflatex`, with no `-shell-escape`.

To add a section or an exercise, edit that file; to start a second collection,
drop another `.tex` in `exercises/` following the same pattern and
`make exercises` picks it up automatically.

---

## 🛠 Compilation Guide

### Prerequisites
Ensure you have a complete TeX Live distribution installed:
```bash
sudo apt-get install texlive-full
```

### Build Instructions using `make`
From the root directory:

- **Build all slides, background notes and exercise sets**:
  ```bash
  make all
  ```

- **Build only the 12 lecture slides**:
  ```bash
  make slides
  ```

- **Build background PDFs**:
  ```bash
  make background
  ```

- **Build the exercise sets** (both the handout and the solutions version of each):
  ```bash
  make exercises
  ```

- **Clean intermediate build files**:
  ```bash
  make clean
  ```

- **Build a specific lecture PDF**:
  ```bash
  make pdf/"01. Introduction.pdf"
  ```
