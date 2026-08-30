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
├── background/                    # Background lectures & supplementary notes
│   ├── lp_background.tex          # Linear Programming fundamentals
│   ├── kkt_background.tex         # KKT optimality conditions
│   ├── supp_material.tex          # Supplementary derivations
│   ├── basis.tex                  # Probability theory background
│   ├── d1.tex                     # Problem exercises
│   └── first_order.tex            # Convex analysis & first-order conditions
├── sty/                           # Custom LaTeX macro definitions and style packages
│   ├── macros.tex                 # Common mathematical vector & matrix shortcuts
│   ├── crayola.sty                # Crayola color palette definitions
│   ├── eclbkbox.sty               # Breakable box environment style
│   └── mathlist.sty               # Custom list environment for math slides
├── imgs/                          # Figure, diagram, and plot assets (EPS, PNG, PDF, JPG)
└── code/                          # Associated code, notebooks, and scripts
    ├── chance constrained toy example.ipynb  # Jupyter notebook for chance-constrained LP
    ├── cp.plot                               # Gnuplot script
    └── hello.jl                              # Julia sample script
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
- [`background/lp_background.tex`](file:///home/bastin/slash/Git/SP_Introduction/background/lp_background.tex) — Linear Programming background
- [`background/kkt_background.tex`](file:///home/bastin/slash/Git/SP_Introduction/background/kkt_background.tex) — Karush-Kuhn-Tucker (KKT) optimality conditions

---

## 🛠 Compilation Guide

### Prerequisites
Ensure you have a complete TeX Live distribution installed:
```bash
sudo apt-get install texlive-full
```

### Build Instructions using `make`
From the root directory:

- **Build all slides and background PDFs**:
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

- **Clean intermediate build files**:
  ```bash
  make clean
  ```

- **Build a specific lecture PDF**:
  ```bash
  make pdf/"01. Introduction.pdf"
  ```
