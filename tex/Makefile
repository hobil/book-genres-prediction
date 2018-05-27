export TEXINPUTS=../tex//:;

all: thesis.pdf
	mv *.aux *.log *.xmpi *.bbl *.blg *.out *.toc build

# LaTeX must be run multiple times to get references right
thesis.pdf: thesis.tex $(wildcard *.tex) bibliography.bib #thesis.xmpdata
	pdflatex $< -output-directory res
	bibtex thesis
	pdflatex $< -output-directory=res
	pdflatex $< -output-directory=res

clean:
	rm -f *.log *.dvi *.aux *.toc *.lof *.lot *.out *.bbl *.blg *.xmpi
	rm -f thesis.pdf
