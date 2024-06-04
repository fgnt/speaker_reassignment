# Once more Diarization: Improving meeting transcription systems through segment-level speaker reassignment

This repository contains the speaker reassignment tool, that was used in the
paper "Once more Diarization: Improving meeting transcription systems through
segment-level speaker reassignment".


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
The `speaker` is may be used, if you use a sticky algorithm, that tries to keep
the speaker labels. If you do not use a sticky algorithm, the speaker labels
are ignored.

You may provide an `emb` and `emb_samples` field,
see [Custom embedding extractor](#custom-embedding-extractor).

The remaining fields are ignored.

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
@inproceedings{Boeddeker2018,
  title={Once more Diarization: Improving meeting transcription systems through segment-level speaker reassignment},
  author={Boeddeker, Christoph and Cord-Landwehr, Tobias and Haeb-Umbach, Reinhold},
  booktitle={Submitted to Interspeech 2024},
  year={2024}
}
```

ToDo: Update the citation once the paper is published.