# Once more Diarization: Improving meeting transcription systems through segment-level speaker reassignment

[![arXiv](https://img.shields.io/badge/arXiv-2406.03155-b31b1b.svg)](https://arxiv.org/abs/2406.03155)


This repository contains the speaker reassignment tool, that was proposed in the
paper "Once more Diarization: Improving meeting transcription systems through
segment-level speaker reassignment". The tool aims to correct speaker confusion errors in a meeting transcription system after a diarization and enhancement.

Please refer to the paper for more information ([Once more Diarization: Improving meeting transcription systems through segment-level speaker reassignment](http://www.arxiv.org/abs/2406.03155)).

# Installation

```bash
pip install git+https://github.com/fgnt/paderbox.git
pip install git+https://github.com/fgnt/padertorch.git
git clone https://github.com/fgnt/speaker_reassignment.git
cd speaker_reassignment
pip install -e .
```

# Usage

For processing, a JSON file `hyp.json` containing the segments (see section "Input format"
later in this readme for the content of this file) is assumed. Then, you can run the reassignments with the following commands:

```bash
python -m speaker_reassignment sc hyp.json  # Just spectral clustering
python -m speaker_reassignment sc_step hyp.json  # Spectral clustering with step-wise attenuation
python -m speaker_reassignment sc_poly hyp.json  # Spectral clustering with polynomial attenuation
python -m speaker_reassignment kmeans hyp.json  # Just k-means
```
Each command will create a new JSON file, e.g. `hyp_SLR_SC_step0.25.json` with the reassigned segments corresponding to the used options.

# Example

For one of our experiments on LibriCSS (TS-SEP + GSS), all necessary files were uploaded.
In [`egs/tssep_gss_wavLMASR`](https://github.com/fgnt/speaker_reassignment/tree/master/egs/tssep_gss_wavLMASR)
you can find a `run.sh` script, that runs the speaker reassignment on the
LibriCSS dataset. The script downloads the enhanced data, the `hyp.json` file
and the `ref.stm` from
[huggingface](https://huggingface.co/datasets/boeddeker/libri_css_tssep_gss_wavLMASR).
It then runs the speaker reassignment for multiple parameterizations and calculates
the cpWER for each of them.
Finally, it prints the cpWER for each speaker reassignment:
```bash
$ cat results.txt 
file                           | error_rate | errors | length  | insertions | deletions | substitutions | missed_speaker | falarm_speaker | scored_speaker
------------------------------ + ---------- + ------ + ------- + ---------- + --------- + ------------- + -------------- + -------------- + --------------
hyp_cpwer.json                 |      5.36% |  5_760 | 107_383 |      1_538 |     2_003 |         2_219 |              0 |              0 |            480
hyp_SLR_C7sticky_cpwer.json    |      5.16% |  5_545 | 107_383 |      1_446 |     1_911 |         2_188 |              0 |              0 |            480
hyp_SLR_kmeans_cpwer.json      |      3.48% |  3_736 | 107_383 |        719 |     1_184 |         1_833 |              0 |              0 |            480
hyp_SLR_SC_cpwer.json          |      3.67% |  3_940 | 107_383 |        792 |     1_257 |         1_891 |              0 |              0 |            480
hyp_SLR_SC_step0.25_cpwer.json |      3.51% |  3_768 | 107_383 |        729 |     1_194 |         1_845 |              0 |              0 |            480
hyp_SLR_SC_poly4_cpwer.json    |      3.50% |  3_763 | 107_383 |        727 |     1_192 |         1_844 |              0 |              0 |            480
```


# Input format

As input, the speaker reassignment tool expects a JSON file (CHiME-5/6/7 style)
with the following structure:

```
[
  {
    "session_id": "overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9",
    "speaker": "6",
    "start_time": 3.0093125,
    "end_time": 7.0093125,
    "audio_path": ".../overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9_6_48149_112149.wav",
    "words": "THE GLIMMERING SEA OF DELICATE LEAVES WHISPERED AND MURMURED BEFORE HER",
    ...
  },
]
```

which is known from CHiME-5/6/7 and called SegLST in [meeteval](https://github.com/fgnt/meeteval).

The `session_id` is used to identify the segments, that belong to the same recoding.
The `audio_path` is used to load the audio and calculate the embedding.
Note: The `audio_path` should point to the audio path of the segment, and not
to the full recording stream. This means, that start and end times are not used
for slicing.
The `speaker` may be used, if you use a sticky algorithm, that tries to keep
the speaker labels. If you do not use a sticky algorithm, the speaker labels
are ignored.

You may provide an `emb` and `emb_samples` field,
see [Custom embedding extractor](#custom-embedding-extractor).

All remaining fields are ignored.

# Custom embedding extractor

If you want to use your own embedding extractor, you can provide the `emb` and
`emb_samples` fields in the JSON file. The `emb` field should contain the
embedding of the segment, and the `emb_samples` field should contain the number
of samples that were used to calculate the embedding.

Alternatively, you can modify the source code to use your own embedding
extractor. Search for 

```
    @functools.cached_property
    def resnet(self):
        # Returns an embedding extractor, that takes the audio as input and
        # returns the embedding.
        # d['emb'] = self.resnet(audio)
        return PretrainedModel(consider_mpi=True)
```

in the `core.py` file and replace the `PretrainedModel` with your own embedding
extractor.

# Cite

If you use this code, please cite the following paper:

```
@inproceedings{Boeddeker2024,
  title={Once more Diarization: Improving meeting transcription systems through segment-level speaker reassignment},
  author={Boeddeker, Christoph and Cord-Landwehr, Tobias and Haeb-Umbach, Reinhold},
  booktitle={Accepted for Interspeech 2024},
  year={2024}
}
```

ToDo: Update the citation once the paper is published.
