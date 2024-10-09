# Analysis

Scripts to analyze the data. First consolidate it, then add metrics,
then plot. To build a plot stored in `metrics.png` from a directory of
results in `/foo/bar/results`, do the following:

```bash
python data-clean.py --data /foo/bar/results \
    | python add-metrics.py \
    | python plot-accuracy.py --output metrics.png
```
