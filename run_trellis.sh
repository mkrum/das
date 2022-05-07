#! /bin/bash

NL=$1

mkdir -p ~/das_data
while true; do
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 64 --DFA.width 2 --model.n_tokens 66 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 64 --DFA.width 4 --model.n_tokens 66 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 64 --DFA.width 8 --model.n_tokens 66 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 64 --DFA.width 16 --model.n_tokens 66 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 64 --DFA.width 32 --model.n_tokens 66 --model.nlayers $NL
    python main.py trellis.yml $(mktemp -p ~/das_data -d) --DFA.alpha_num 64 --DFA.width 64 --model.n_tokens 66 --model.nlayers $NL
done
