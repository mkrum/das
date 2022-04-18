#! /bin/bash

NL=$1

while true; do
    python main.py config.yml $(mktemp -p /nfs/das_data -d) --DFA.n_states 4 --model.nlayers $NL
    python main.py config.yml $(mktemp -p /nfs/das_data -d) --DFA.n_states 6 --model.nlayers $NL
    python main.py config.yml $(mktemp -p /nfs/das_data -d) --DFA.n_states 8 --model.nlayers $NL
    python main.py config.yml $(mktemp -p /nfs/das_data -d) --DFA.n_states 10 --model.nlayers $NL
done
