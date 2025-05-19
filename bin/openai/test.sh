#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

source $ROOT/config.rc
source $ROOT/venv/bin/activate

_src=$ROOT/models/open-ai-finetune
_var=$SMALL_TALK_OUTPUT/openai

output=$_var/results
rm --recursive --force $output
mkdir --parents $output

for i in $_var/models/*.json; do
    out=$output/`basename --suffix=.json $i`.csv
    if [ ! -e $out ]; then
	cat <<EOF
echo "[ `date` ] $i" 1>&2 \
    && python $_src/test.py \
	      --system-prompt $_src/system.txt \
	      --data $SMALL_TALK_DATA < $i > $out
EOF
    fi
done | parallel --will-cite --line-buffer
