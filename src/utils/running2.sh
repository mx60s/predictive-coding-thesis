#!/bin/bash

for i in {1..12}
do
	python3 walk.py --steps 5000 --port 9000

	echo "Iteration number: $i"

done
