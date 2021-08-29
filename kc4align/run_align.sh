#!/bin/sh
IFS='/'
read -a strarr <<< "$1"
screen -XS ${strarr[-1]} quit
if [ -z "$STY" ]; then exec screen -dm -S ${strarr[-1]} /bin/bash activate_align.sh $1 $2; fi
