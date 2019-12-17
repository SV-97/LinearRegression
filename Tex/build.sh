#!/bin/bash
xelatex -shell-escape test
bibtex test
xelatex -shell-escape test
xelatex -shell-escape test
