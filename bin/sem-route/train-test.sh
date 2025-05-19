#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

source $ROOT/config.rc
source $ROOT/venv/bin/activate

output=$SMALL_TALK_OUTPUT/semantic-router/results
rm --recursive --force $output
mkdir --parents $output

for i in $SMALL_TALK_DATA/*.csv; do
    out=$output/`basename $i`
    cat <<EOF
echo "[ `date` ] $i" 1>&2 && \
    python $ROOT/models/semantic-router/test.py --data $i \
    	| grep --invert-match '^sagemaker.config' > $out
EOF
done | parallel --will-cite --line-buffer
