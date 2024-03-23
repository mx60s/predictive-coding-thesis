#!/bin/zsh

for i in {1..12}
do
   echo "Run #$i"
   python3 continuous-forward.py
done

echo "Completed 12 runs."

