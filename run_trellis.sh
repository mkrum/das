#! /bin/bash

NL=$1

mkdir -p ~/das_data
while true; do
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 2 --model.n_tokens 10 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 4 --model.n_tokens 10 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 8 --model.n_tokens 10 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 16 --model.n_tokens 18 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 32 --model.n_tokens 34 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 64 --model.n_tokens 66 --model.nlayers $NL
done
