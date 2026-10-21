# Auxiliary files in build/, main.pdf next to main.tex.
@default_files = ('main.tex');
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';
$aux_dir = 'build';
$out_dir = '.';
# Project style files live in style/.
ensure_path('TEXINPUTS', './style//');
# BibTeX runs inside build/, so look for refs.bib one level up.
ensure_path('BIBINPUTS', '..', '.');
