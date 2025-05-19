#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

source $ROOT/config.rc
source $ROOT/venv/bin/activate

_tr_fracs=( `seq 10 15 99` )
_seeds=10

tmp=`mktemp`
results=$ROOT/var/binary_c/data

mkdir --parents $results
python $ROOT/train-test-split/build-splits.py \
       --collapse-negatives < sneha.csv > $tmp
for i in ${_tr_fracs[@]}; do
    tsize=$(bc -l <<< "$i / 100")
    for j in `seq $_seeds`; do
	fname=`uuid -v4`
	cat <<EOF
echo "[ `date` ] $i $j" 1>&2 && \
    python $ROOT/train-test-split/refine-train.py \
	   --seed $RANDOM \
	   --train-size $tsize < $tmp > $results/$fname.csv
EOF
    done
done | parallel --will-cite --line-buffer
rm $tmp
