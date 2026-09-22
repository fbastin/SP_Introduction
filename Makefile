# Makefile for Stochastic Programming Introduction Course

SLIDES_DIR = slides
BACKGROUND_DIR = background
EXERCISES_DIR = exercises
PDF_DIR = pdf
STY_DIR = sty
IMGS_DIR = imgs

# Search paths for LaTeX
export TEXINPUTS := .//:./$(SLIDES_DIR)//:./$(BACKGROUND_DIR)//:./$(EXERCISES_DIR)//:./$(STY_DIR)//:./$(IMGS_DIR)//:../sty//:../imgs//:$(TEXINPUTS)

# No -shell-escape: PSTricks/auto-pst-pdf are gone (pictures are TikZ now) and
# the `graphs` target pre-converts every EPS figure.
PDFLATEX = pdflatex -interaction=nonstopmode

# Filenames contain spaces (e.g. "01. Introduction.tex"), which GNU make
# cannot handle in wildcard lists: word-splitting ignores escaped spaces.
# The `slides` target therefore loops over the sources in the shell and
# delegates each PDF to the pattern rule below.
SLIDES_GLOB = $(SLIDES_DIR)/*.tex

# `plan.tex` does not exist yet: wildcard keeps only the sources actually present,
# so `make background` works now and picks it up automatically if it is added later.
STANDALONE_BACKGROUND = $(wildcard $(BACKGROUND_DIR)/lp_background.tex $(BACKGROUND_DIR)/kkt_background.tex $(BACKGROUND_DIR)/probability_background.tex $(BACKGROUND_DIR)/plan.tex)
BACKGROUND_PDF = $(patsubst $(BACKGROUND_DIR)/%.tex,$(PDF_DIR)/%.pdf,$(STANDALONE_BACKGROUND))

.PHONY: all slides background exercises graphs clean help

all: slides background exercises

# epstopdf does not resolve inputs through TEXINPUTS on all installations:
# pre-convert every EPS figure next to its source so that pdflatex finds
# the <name>-eps-converted-to.pdf files via kpathsea (no shell-escape needed).
graphs:
	cd $(IMGS_DIR) && for f in *.eps; do \
		[ -f "$${f%.eps}-eps-converted-to.pdf" ] || \
		epstopdf --outfile="$${f%.eps}-eps-converted-to.pdf" "$$f"; \
	done

slides: graphs
	@mkdir -p $(PDF_DIR)
	@for f in $(SLIDES_GLOB); do \
		b=$$(basename "$$f" .tex); \
		$(MAKE) --no-print-directory "$(PDF_DIR)/$$b.pdf" || exit 1; \
	done

background: $(BACKGROUND_PDF)

# Each exercise set is a single source yielding two PDFs: the student version
# (statements only) and the version with solutions. `\nosolutions` is defined on
# the command line, so there is nothing to keep in sync between the two.
exercises:
	@mkdir -p $(PDF_DIR)
	@for f in $(EXERCISES_DIR)/*.tex; do \
		[ -e "$$f" ] || continue; \
		b=$$(basename "$$f" .tex); \
		(cd $(EXERCISES_DIR) && \
		 $(PDFLATEX) -jobname="$${b}_solutions" "$$b.tex" && \
		 $(PDFLATEX) -jobname="$${b}_solutions" "$$b.tex" && \
		 $(PDFLATEX) -jobname="$$b" '\def\nosolutions{}\input{'"$$b"'.tex}' && \
		 $(PDFLATEX) -jobname="$$b" '\def\nosolutions{}\input{'"$$b"'.tex}') || exit 1; \
		mv "$(EXERCISES_DIR)/$${b}_solutions.pdf" "$(PDF_DIR)/"; \
		mv "$(EXERCISES_DIR)/$$b.pdf" "$(PDF_DIR)/"; \
	done

$(PDF_DIR)/%.pdf: $(SLIDES_DIR)/%.tex
	@mkdir -p $(PDF_DIR)
	jn=$$(printf '%s' '$*' | tr -d ' '); \
	(cd $(SLIDES_DIR) && $(PDFLATEX) -jobname="$$jn" "$*.tex") && \
	(cd $(SLIDES_DIR) && $(PDFLATEX) -jobname="$$jn" "$*.tex") && \
	mv "$(SLIDES_DIR)/$$jn.pdf" "$@"

$(PDF_DIR)/%.pdf: $(BACKGROUND_DIR)/%.tex
	@mkdir -p $(PDF_DIR)
	cd $(BACKGROUND_DIR) && $(PDFLATEX) "$(notdir $<)"
	cd $(BACKGROUND_DIR) && $(PDFLATEX) "$(notdir $<)"
	mv $(BACKGROUND_DIR)/$(notdir $@) $@

clean:
	@echo "Cleaning TeX build artifacts..."
	@rm -f $(SLIDES_DIR)/*.aux $(SLIDES_DIR)/*.log $(SLIDES_DIR)/*.nav $(SLIDES_DIR)/*.out $(SLIDES_DIR)/*.snm $(SLIDES_DIR)/*.toc $(SLIDES_DIR)/*.vrb $(SLIDES_DIR)/*.fls $(SLIDES_DIR)/*.fdb_latexmk $(SLIDES_DIR)/*-eps-converted-to.pdf
	@rm -f $(BACKGROUND_DIR)/*.aux $(BACKGROUND_DIR)/*.log $(BACKGROUND_DIR)/*.nav $(BACKGROUND_DIR)/*.out $(BACKGROUND_DIR)/*.snm $(BACKGROUND_DIR)/*.toc $(BACKGROUND_DIR)/*.vrb $(BACKGROUND_DIR)/*.fls $(BACKGROUND_DIR)/*.fdb_latexmk $(BACKGROUND_DIR)/*-eps-converted-to.pdf
	@find $(EXERCISES_DIR) -type f \( -name '*.aux' -o -name '*.log' -o -name '*.nav' \
	  -o -name '*.out' -o -name '*.snm' -o -name '*.toc' -o -name '*.vrb' \
	  -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.synctex.gz' \) -delete
	@rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk

help:
	@echo "Stochastic Programming Course Makefile"
	@echo "Commands:"
	@echo "  make all        - Build all lecture slides and background PDFs"
	@echo "  make slides     - Build all 12 lecture slide PDFs"
	@echo "  make background - Build background lecture PDFs"
	@echo "  make exercises  - Build each exercise set twice: statements only, and with solutions"
	@echo "  make clean      - Clean LaTeX intermediate build artifacts"
