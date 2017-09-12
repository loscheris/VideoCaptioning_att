#!/usr/bin/env bash

for i in {1..300}
do
files=(./rgb_feats/*)
mv ${files[RANDOM % ${#files[@]}]} ./rgb_test_features/
done
