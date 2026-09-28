# Build the manuscript PDF and the arXiv source bundle.
#
#   make          build main.pdf (tectonic runs TeX/BibTeX passes as needed)
#   make arxiv    build arxiv/ (self-contained source: figures copied, the
#                 .bbl input in place of \bibliography, graphicspath
#                 rewritten), test-compile it, and pack it as arxiv.tar.gz
#
# Numbers in the paper come from generated/*.tex, written by
# `uv run python -m experiment.run` from the repository root.

TECTONIC = tectonic
MAIN     = main
FIGS     = policy_evolution forgetting transfer judge_reliability

.PHONY: all arxiv clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib generated/*.tex
	$(TECTONIC) -X compile $(MAIN).tex --keep-intermediates
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).blg

arxiv: $(MAIN).pdf
	rm -rf arxiv arxiv.tar.gz
	mkdir -p arxiv/figures arxiv/generated
	cp generated/*.tex arxiv/generated/
	for f in $(FIGS); do cp ../experiment/figures/$$f.pdf arxiv/figures/; done
	sed -e 's|\\graphicspath{{../experiment/figures/}}|\\graphicspath{{figures/}}|' \
	    -e 's|\\bibliography{references}|\\input{$(MAIN).bbl}|' $(MAIN).tex > arxiv/$(MAIN).tex
	cp $(MAIN).bbl arxiv/$(MAIN).bbl
	check=$$(mktemp -d) && cp -R arxiv/. $$check/ && (cd $$check && $(TECTONIC) -X compile $(MAIN).tex >/dev/null) && echo "arXiv bundle compiles" && rm -rf $$check
	tar -czf arxiv.tar.gz -C arxiv .

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).blg $(MAIN).bbl
	rm -rf arxiv arxiv.tar.gz
