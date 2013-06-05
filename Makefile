# ==============================================================
#
#    Makefile (Top Level)
#
# --------------------------------------------------------------
# Copyright (c) Carnegie Mellon University, 2012-
#
# Description: 
#
#   Makefile rules to compile the project under Linux
#
# Revisions:
#
#  ============================================================


DIRS	:= src

MAKE=make
SHELL=/bin/bash

.PHONY: $(DIRS) all clean

all: $(DIRS)

# Make all subdirectories specified for MAKE
$(DIRS):
	@if [[ -d $@ ]]; then \
		echo Running make in $@; \
		cd $@; \
		$(MAKE); \
	 else \
		echo Directory not found: $@. Skipping...; \
	 fi;

# Recursive clean rule
clean: 
	@if [ "x$(DIRS)" != "x" ]; then \
	set $(DIRS); \
	for x ; do \
		echo Cleaning $$x; \
		$(MAKE) -C $$x clean; \
	done; fi
