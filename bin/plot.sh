#!/bin/bash

tmp=`mktemp`

python analysis/data-clean.py --data results/binary_c > $tmp

mc=mis-class
python analysis/$mc.py \
       --output $mc.png \
       --dump-raw $mc.csv \
       --without-zero < $tmp

# python analysis/add-metrics.py \
#     | python analysis/plot-metrics.py --output metrics.png < $tmp

rm $tmp
