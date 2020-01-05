#!/bin/sh
find data -type f -name '*.bz2' | parallel bzip2 -d '{}'
cd data
find . -type f -name '*.tar' | parallel tar -xf '{}'
find data -type f -name '*.tar' -exec rm -rf '{}' \;