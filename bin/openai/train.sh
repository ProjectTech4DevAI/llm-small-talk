#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

source $ROOT/config.rc
source $ROOT/venv/bin/activate

_src=$ROOT/models/open-ai-finetune
_var=$ROOT/var

ft_out=$_var/binary_c/openai-ft
# rm --recursive --force $ft_out
# mkdir --parents $ft_out

python $_src/select-data.py --data $_var/binary_c/data \
    | while read; do
    out=$ft_out/`basename --suffix=.csv $REPLY`.json
    if [ ! -e $out ]; then
	tmp=`mktemp`
	cat <<EOF
echo "[ `date` ] $REPLY" 1>&2 \
    && python $_src/train.py \
              --system-prompt $_src/system.txt \
              --wait-time-minutes 6 \
              --data $REPLY > $tmp \
    && mv $tmp $out
EOF
    fi
done | parallel \
	   --will-cite \
	   --line-buffer \
	   --max-procs 3 \
	   --delay 2 \
	   --halt soon,fail=1
