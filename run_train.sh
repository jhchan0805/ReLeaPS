#!/bin/bash

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
make data/models/20/{ls,cnnps,psfcn}.bin

