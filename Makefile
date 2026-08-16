# Makefile for Stochastic Programming Introduction Course

SLIDES_DIR = slides
BACKGROUND_DIR = background
PDF_DIR = pdf
STY_DIR = sty
IMGS_DIR = imgs

# Search paths for LaTeX
export TEXINPUTS := .//:./$(SLIDES_DIR)//:./$(BACKGROUND_DIR)//:./$(STY_DIR)//:./$(IMGS_DIR)//:../sty//:../imgs//:$(TEXINPUTS)

PDFLATEX = pdflatex -interaction=nonstopmode -shell-escape

SLIDES_SRC = $(wildcard $(SLIDES_DIR)/*.tex)
SLIDES_PDF = $(patsubst $(SLIDES_DIR)/%.tex,$(PDF_DIR)/%.pdf,$(SLIDES_SRC))

STANDALONE_BACKGROUND = $(BACKGROUND_DIR)/lp_background.tex $(BACKGROUND_DIR)/kkt_background.tex $(BACKGROUND_DIR)/plan.tex
BACKGROUND_PDF = $(patsubst $(BACKGROUND_DIR)/%.tex,$(PDF_DIR)/%.pdf,$(STANDALONE_BACKGROUND))

.PHONY: all slides background clean help

all: slides background

slides: $(SLIDES_PDF)

background: $(BACKGROUND_PDF)

$(PDF_DIR)/%.pdf: $(SLIDES_DIR)/%.tex
	@mkdir -p $(PDF_DIR)
	cd $(SLIDES_DIR) && $(PDFLATEX) "$(notdir $<)"
	cd $(SLIDES_DIR) && $(PDFLATEX) "$(notdir $<)"
	mv $(SLIDES_DIR)/$(notdir $@) $@

$(PDF_DIR)/%.pdf: $(BACKGROUND_DIR)/%.tex
	@mkdir -p $(PDF_DIR)
	cd $(BACKGROUND_DIR) && $(PDFLATEX) "$(notdir $<)"
	cd $(BACKGROUND_DIR) && $(PDFLATEX) "$(notdir $<)"
	mv $(BACKGROUND_DIR)/$(notdir $@) $@

clean:
	@echo "Cleaning TeX build artifacts..."
	@rm -f $(SLIDES_DIR)/*.aux $(SLIDES_DIR)/*.log $(SLIDES_DIR)/*.nav $(SLIDES_DIR)/*.out $(SLIDES_DIR)/*.snm $(SLIDES_DIR)/*.toc $(SLIDES_DIR)/*.vrb $(SLIDES_DIR)/*.fls $(SLIDES_DIR)/*.fdb_latexmk $(SLIDES_DIR)/*-eps-converted-to.pdf
	@rm -f $(BACKGROUND_DIR)/*.aux $(BACKGROUND_DIR)/*.log $(BACKGROUND_DIR)/*.nav $(BACKGROUND_DIR)/*.out $(BACKGROUND_DIR)/*.snm $(BACKGROUND_DIR)/*.toc $(BACKGROUND_DIR)/*.vrb $(BACKGROUND_DIR)/*.fls $(BACKGROUND_DIR)/*.fdb_latexmk $(BACKGROUND_DIR)/*-eps-converted-to.pdf
	@rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk

help:
	@echo "Stochastic Programming Course Makefile"
	@echo "Commands:"
	@echo "  make all        - Build all lecture slides and background PDFs"
	@echo "  make slides     - Build all 12 lecture slide PDFs"
	@echo "  make background - Build background lecture PDFs"
	@echo "  make clean      - Clean LaTeX intermediate build artifacts"
