#!/bin/bash

#Used for generating .scores files for all the cases
#It will run make_scores_file.py
#Input the list of case name
#Usage: all_scores.sh <list>
#Date: 26-04-2018, Author: Zhengyue Zhang

list=`more $1`

for item in $list
do
	python $MYSCRIPTS/make_scores_file.py $item
done
