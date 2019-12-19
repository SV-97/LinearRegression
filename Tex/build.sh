#!/bin/bash
xelatex -shell-escape paper
bibtex paper
xelatex -shell-escape paper
xelatex -shell-escape paper
