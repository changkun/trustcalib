# Makefile for compiling the manuscript PDF.
# `make` builds main.pdf (runs pdflatex/bibtex as many passes as needed via latexmk).

LATEXMK = latexmk
MAIN    = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	$(LATEXMK) -C
	rm -f $(MAIN).bbl
