#!/bin/bash

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
make data/benchmark/{blobs,sculpture,diligent}_{all,rand_20,rl_20_test}.txt

