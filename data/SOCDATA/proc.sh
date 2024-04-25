#!/bin/bash

FILE=$1
OFILE='data_'$FILE
n=`grep -n date $FILE | awk -F":" '{print $1}'`
n=$((n-1))

sed -e "1,${n}d" $FILE > $OFILE
