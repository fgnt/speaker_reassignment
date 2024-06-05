#!/usr/bin/env bash

# Exit on error
set -e

# Print the commands being run
set -x


python download_and_prepare.py

# This first call will be slow, as it will calculate the embeddings for all the audio files.
# You can prefix the command with `mpiexec -np 8` to speed it up. Replace 8 with the number of cores you have.
python -m speaker_reassignment sc hyp.json
python -m speaker_reassignment sc_step hyp.json
python -m speaker_reassignment sc_poly hyp.json
python -m speaker_reassignment c7sticky hyp.json
python -m speaker_reassignment kmeans hyp.json

meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp.json
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_C7sticky.json
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_kmeans.json
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_SC.json
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_SC_step0.25.json
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_SC_poly4.json

python results_to_table.py > results.txt
cat results.txt

