#! /bin/bash

NL=$1

mkdir -p ~/das_data
while true; do
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --model.nlayers $NL
done
