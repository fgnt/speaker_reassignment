#!/usr/bin/env bash

# Print the commands being run
set -x

# Exit on error
set -e

python download_and_prepare.py

# This first call will be slow, as it will calculate the embeddings for all the audio files.
python -m speaker_reassignment sc hyp.json
python -m speaker_reassignment sc_step hyp.json
python -m speaker_reassignment sc_poly hyp.json
python -m speaker_reassignment c7sticky hyp.json
python -m speaker_reassignment kmeans hyp.json

meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp.json --per-reco-out /dev/null --average-out - | grep error_rate
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_SC.json --per-reco-out /dev/null --average-out - | grep error_rate
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_SC_step0.25.json --per-reco-out /dev/null --average-out - | grep error_rate
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_SC_poly4.json --per-reco-out /dev/null --average-out - | grep error_rate
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_C7sticky.json --per-reco-out /dev/null --average-out - | grep error_rate
meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r ref.stm -h hyp_SLR_kmeans.json --per-reco-out /dev/null --average-out - | grep error_rate