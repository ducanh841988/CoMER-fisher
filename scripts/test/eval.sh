#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=4 ./scripts/test/eval.sh $version $test_split $error_tol
# test_split: val, 2019, 2023 (folder under preprocessing/output/dataset/data/)


version=$1
test_split=$2
error_tol=$3
dir=lightning_logs/version_$version/
test_path="preprocessing/output/dataset/data/${test_split}"

# clean out
rm -rf test_temp/
rm -rf Results_pred_symlg/

# generate predictions
python scripts/test/test.py $version "$test_path"

# copy predictions to target folder
cp result.zip $dir/$test_split.zip

# dump predictions to temp
mkdir -p test_temp/result
unzip -q result.zip -d test_temp/result

# convert tex to symlg
tex2symlg test_temp/result test_temp/pred_symlg

# evaluate two symlg folder
evaluate test_temp/pred_symlg data/$test_split/symLg >/dev/null 2>&1

# extract evaluation result and save to target folder
python scripts/test/extract_exprate.py $error_tol >&1 | tee $dir/$test_split.txt
