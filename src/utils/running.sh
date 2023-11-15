#!/bin/bash

for i in {1..100000}
do
	python3 walk.py --steps 100000

	echo "Iteration number: $i"

done
