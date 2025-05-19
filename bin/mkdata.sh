#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

source $ROOT/config.rc
source $ROOT/venv/bin/activate

_seeds=10

while getopts 'f:d:s:h' option; do
    case $option in
        f) _tr_fracs=( ${_tr_fracs[@]} $OPTARG ) ;;
	d) _data=$OPTARG ;;
	s) _seeds=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -f Train size fraction (between 0 and 100). Can be specified multiple
    times
 -d Input data (CSV)
 -s Number of seeds (training runs) to build per train size fraction
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

if [ ${#_tr_fracs[@]} -eq 0 ]; then
    _tr_fracs=( `seq 10 15 99` )
fi


tmp=`mktemp`
python $ROOT/train-test-split/build-splits.py \
       --collapse-negatives < $_data > $tmp

mkdir --parents $SMALL_TALK_DATA
for i in ${_tr_fracs[@]}; do
    tsize=$(bc -l <<< "$i / 100")
    for j in `seq $_seeds`; do
	fname=`uuid -v4`
	cat <<EOF
echo "[ `date` ] $i $j" 1>&2 && \
    python $ROOT/train-test-split/refine-train.py \
	   --seed $RANDOM \
	   --train-size $tsize < $tmp > $SMALL_TALK_DATA/$fname.csv
EOF
    done
done | parallel --will-cite --line-buffer

rm $tmp
