#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

source $ROOT/config.rc
source $ROOT/venv/bin/activate

_src=$ROOT/models/open-ai-finetune

output=$SMALL_TALK_OUTPUT/openai/models
rm --recursive --force $output
mkdir --parents $output

python $_src/select-data.py --data $SMALL_TALK_DATA \
    | while read; do
    out=$output/`basename --suffix=.csv $REPLY`.json
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
