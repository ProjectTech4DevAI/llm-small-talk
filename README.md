# LLM Small Talk Classification

This repository contains code that classifies input from a
chatbot. Classification is important in chatbots that are designed to
answer questions; particularly when those questions are passed along
to an LLM. Being able to distinguish between actual questions and
irrelevant input helps the chatbot be smarter about what it sends to
the LLM, and the LLM in turn to be more consistent with its answers.

More information on the small talk project, the project for which this
code was designed, can be found
[here](https://docs.google.com/document/d/1q7GWzrLkTf4lrTYQsXjzUrnfuQ-nLzw7fjH16Zr9cSI/edit?usp=sharing).

# Background

## Data

This codebase starts from the assumption that there is a CSV file
containing columns _Question_, and _Classification_. Values in the
question column are input values from users. Their corresponding
values in the classification column are how they should be
categorized.

When building small talk, classifications took on the following
values:

* acknowledgement
* ignore
* mistake
* other
* query
* salutation
* spam

Questions classified as "ignore" were not used. Questions classified
as "query" were kept as is. All other questions were given a new
classification of "small-talk". See `train-test-split/build-splits.py`
for details.

If your data contains different names for the question and
classification column names, or has classification values that do not
contain "ignore" or "query", this codebase may not work for you
without changes.

## Modelling

There are two two types of modelling this repository considers:

1. Semantic routing via the [semantic-router](https://github.com/aurelio-labs/semantic-router) package.
2. OpenAI fine-tuning via the [OpenAI API](https://platform.openai.com/docs/api-reference/fine-tuning).

# Run

## Environment

The code assumes your environment is set properly:

```bash
ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT
export PYTHONLOGLEVEL=info
export OPENAI_API_KEY=...

SMALL_TALK_OUTPUT=$ROOT/var
SMALL_TALK_DATA=$SMALL_TALK_OUTPUT/data
```

and that the `OUTPUT` directory exists:

```bash
mkdir $OUTPUT
```

## Automated running

Bash scripts in `bin` can be used to run all steps outlined in this
section. Assuming you have run the following commands from the root of
this directory:

```bash
cat <<EOF > config.rc
ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT
export PYTHONLOGLEVEL=info
export OPENAI_API_KEY=...
SMALL_TALK_OUTPUT=$ROOT/var
SMALL_TALK_DATA=$SMALL_TALK_OUTPUT/data
EOF
python -m venv venv
source venv
pip install --requirement requirements.txt
```

You can then run the scripts in `bin`:

```bash
./bin/make-data.sh -d /path/to/question-answers.csv
./bin/sem-route/train-test.sh
./bin/openai/train.sh
./bin/openai/test.sh
```

This will leave results files in `$SMALL_TALK_OUTPUT`:

```bash
find $SMALL_TALK_OUTPUT -name results
```

Those files can be analyzed using scripts in `analysis`.


### Detailed runs

Using the pre-made scripts in `bin` is the recommended way of running
this pipeline. The instructions presented in this section are for
those interested in going behind the scenes.

#### Data preparation

Create a temporary directory from which to work:

```bash
tmp=`mktemp`
```

The first step is to binarize your data, then split it into train and
test sets:

```bash
python train-test-split/build-splits.py \
       --collapse-negatives < /path/to/question-answers.csv > $tmp
```

To produce robust results, several models are trained with different
versions of the training set. To do this further split the training
set into multiple smaller train and test sets. Several subsets are
created that differ in size, and the random choice of question-answer
pairs that are selected:

```bash
for i in `seq 10 15 99`; do
    tsize=$(bc -l <<< "$i / 100")
    for j in `seq 10`; do
        fname=`uuid -v4`
	python train-test-split/refine-train.py \
	       --seed $RANDOM \
	       --train-size $tsize < $tmp > $SMALL_TALK_DATA/$fname.csv
    done
done
```

Clean up
```bash
rm $tmp
```

#### Semantic Router

Run the semantic router using the all of the training sets created:
The framework prints an odd log message regarding "sagemaker" to
stdout that needs to be captured before its results are saved:

```bash
output=$SMALL_TALK_OUTPUT/semantic-router/results
mkdir --parents $output
for i in $SMALL_TALK_DATA/*.csv; do
    out=$output/`basename $i`
    python models/semantic-router/test.py --data $i \
        | grep --invert-match '^sagemaker.config' > $out
done
```

#### OpenAI Fine-tuning

Fine tuning OpenAI involves two steps. First select the training data
from the train-data subset, then use the OpenAI API to build a model:

```bash
python models/open-ai-finetune/select-data.py --data $SMALL_TALK_DATA \
    | while read; do
    out=$SMALL_TALK_OUTPUT/openai/models/`basename --suffix=.csv $REPLY`.json
    if [ ! -e $out ]; then
        tmp=`mktemp`
	python models/open-ai-finetune/train.py \
	       --system-prompt models/open-ai-finetune/system.txt \
	       --data $REPLY > $tmp \
	    && mv $tmp $out
    fi
done
```

Using the output of each training run -- invocation of `train.py` --
test the fine tuned model on the test set:

```bash
for i in $SMALL_TALK_OUTPUT/openai/models/*.json; do
    out=$SMALL_TALK_OUTPUT/openai/results/`basename --suffix=.json $i`.csv
    if [ ! -e $out ]; then
        python $_src/test.py \
	       --system-prompt models/open-ai-finetune/system.txt \
	       --data $SMALL_TALK_DATA < $i > $out
    fi
done
```

# Analysis

The output from both modelling types are a collection of CSVs
corresponding to each experiment created in the data preparation
phase. The CSVs have the following structure:

| Column | Description |
|--- |---
| data | Data file (CSV) used to generate the result |
| train_n | Number of training examples |
| train_c | Number of training classification types |
| query | User query |
| gt | Ground truth classification |
| split | Split of which the example is a member |
| seed | Random seed (state) |
| pr | Predicted classification |
| duration | Length of time to produce result (seconds) |

There are scripts in `analysis` that can help with parsing and
plotting results in this structure.
