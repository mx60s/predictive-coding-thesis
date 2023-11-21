#!/bin/bash

for i in {1..1000}
do
	python3 walk.py --steps 100000

	echo "Iteration number: $i"

done
