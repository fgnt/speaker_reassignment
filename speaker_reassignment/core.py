"""
python -m speaker_reassignment sc c7.json
"""
import functools
import sys
from pathlib import Path

import numpy as np

import dlp_mpi.collection
import paderbox as pb

from speaker_reassignment.tcl_pretrained import PretrainedModel


class c:
    Red = '\033[91m'
    Color_Off = '\033[0m'


def dist(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    return 1 - np.einsum('...d,...d->...', a, b)


def similarity(emb, emb_samples, scale, version=1):
    """
    >>> emb_samples = np.linspace(0, 8 * 16000, 17)
    >>> emb_samples / 16000
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
           6.5, 7. , 7.5, 8. ])
    >>> emb = np.ones([len(emb_samples), 2])
    >>> for samples, mask in zip(emb_samples, similarity(emb, emb_samples, 0.1, version=1)[0]):
    ...    print(f'{samples / 16000}: {mask}')
    version: 1
    Mask thresh: 0.1
    0.0: 0.0001
    0.5: 0.0001
    1.0: 0.001
    1.5: 0.001
    2.0: 0.01
    2.5: 0.01
    3.0: 0.01
    3.5: 0.01
    4.0: 0.09999999999999998
    4.5: 0.09999999999999998
    5.0: 0.09999999999999998
    5.5: 0.09999999999999998
    6.0: 0.09999999999999998
    6.5: 0.09999999999999998
    7.0: 0.09999999999999998
    7.5: 0.09999999999999998
    8.0: 0.9999999999999998
    >>> for samples, mask in zip(emb_samples, similarity(emb, emb_samples, 1, version=3)[0]):
    ...    print(f'{samples / 16000}: {mask}')
    version: 3
    Mask thresh: 1
    0.0: 0.0
    0.5: 0.062499999999999986
    1.0: 0.12499999999999997
    1.5: 0.18749999999999994
    2.0: 0.24999999999999994
    2.5: 0.31249999999999994
    3.0: 0.3749999999999999
    3.5: 0.4374999999999999
    4.0: 0.4999999999999999
    4.5: 0.5624999999999999
    5.0: 0.6249999999999999
    5.5: 0.6874999999999999
    6.0: 0.7499999999999998
    6.5: 0.8124999999999998
    7.0: 0.8749999999999998
    7.5: 0.9374999999999998
    8.0: 0.9999999999999998
    >>> for samples, mask in zip(emb_samples, similarity(emb, emb_samples, 2, version=3)[0]):
    ...    print(f'{samples / 16000}: {mask}')
    version: 3
    Mask thresh: 2
    0.0: 0.0
    0.5: 0.003906249999999999
    1.0: 0.015624999999999997
    1.5: 0.03515624999999999
    2.0: 0.062499999999999986
    2.5: 0.09765624999999997
    3.0: 0.14062499999999997
    3.5: 0.19140624999999994
    4.0: 0.24999999999999994
    4.5: 0.31640624999999994
    5.0: 0.3906249999999999
    5.5: 0.4726562499999999
    6.0: 0.5624999999999999
    6.5: 0.6601562499999999
    7.0: 0.7656249999999998
    7.5: 0.8789062499999998
    8.0: 0.9999999999999998

    """
    a_length = emb_samples[:, None]
    a = emb[:, None, :]

    b_length = emb_samples[None, :]
    b = emb[None, :, :]

    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    sample_rate = 16000

    print('version:', version)
    if version == 0:
        mask = 1
    elif version == 1:
        # This is called "Step-wise attenuation" in "Once more Diarization:
        # Improving meeting transcription systems through
        # segment-level speaker reassignment"
        mask1 = np.maximum(a_length, b_length) >= sample_rate * 1
        mask2 = np.maximum(a_length, b_length) >= sample_rate * 2
        mask3 = np.maximum(a_length, b_length) >= sample_rate * 4
        mask4 = np.maximum(a_length, b_length) >= sample_rate * 8

        mask1 = np.maximum(mask1, scale)
        mask2 = np.maximum(mask2, scale)
        mask3 = np.maximum(mask3, scale)
        mask4 = np.maximum(mask4, scale)

        print(f'Mask thresh: {scale}')
        mask = mask1 * mask2 * mask3 * mask4
    elif version == 2:
        mask1 = np.maximum(a_length, b_length) >= sample_rate * 1
        mask2 = np.maximum(a_length, b_length) >= sample_rate * 2
        mask3 = np.maximum(a_length, b_length) >= sample_rate * 4

        mask1 = np.maximum(mask1, scale)
        mask2 = np.maximum(mask2, scale)
        mask3 = np.maximum(mask3, scale)

        print(f'Mask thresh: {scale}')
        mask = mask1 * mask2 * mask3
    elif version == 3:
        thr = sample_rate * 8

        print(f'Mask thresh: {scale}')
        mask = np.minimum(1, (np.maximum(a_length, b_length) / thr) ** scale)
    else:
        raise ValueError(version)

    s = np.einsum('...d,...d,...->...', a, b, mask)
    if np.isfinite(s).all():
        return abs(s)
    else:
        raise ValueError(a, b, s)


def sbatch(argv):
    """
    python -m speaker_reassignment sbatch c7.json
    """
    from tssep_data.util.slurm import cmd_to_hpc
    from tssep_data.util.cmd_runner import run

    cmd = cmd_to_hpc(
        ['python', '-m', 'speaker_reassignment', *argv],
        mpi=20, block=False, time='4h'
    )
    run(cmd)


class Prepare:
    def __init__(self):
        self.dump = True

    @functools.cached_property
    def resnet(self):
        # Returns an embedding extractor, that takes the audio as input and
        # returns the embedding.
        # d['emb'] = self.resnet(audio)
        return PretrainedModel(consider_mpi=True)

    def json_to_per_reco(self, json, audio_path, num_speakers):
        """

         - Load json.
         - Compute embeddings if not present and not cached.
         - group by session_id
            - yield session_id, data_rec, num_speakers

        data_rec:
           One entry in the json, with additional entries 'emb' and 'emb_samples'.

        """
        json = Path(json)
        data = dlp_mpi.call_on_root_and_broadcast(pb.io.load, json)

        data_per_reco = pb.utils.iterable.groupby(
            data, lambda x: x['session_id']
        )
        dump_dir = json.with_suffix('').with_stem(json.stem + '_emb')

        statistics = {
            'calculated': 0,
            'loaded_cached': 0,
            'user_provided': 0,
        }
        if self.dump and dlp_mpi.IS_MASTER:
            dump_dir.mkdir(exist_ok=True, parents=True)

        for session_id, data_rec in dlp_mpi.split_managed(
                sorted(data_per_reco.items()), allow_single_worker=True):
            print(f'Process {session_id}')

            if 'emb' in data_rec[0]:
                statistics['user_provided'] += 1
                assert all('emb' in d for d in data_rec), data_rec
                assert all('emb_samples' in d for d in data_rec), data_rec
            elif self.dump and (dump_dir / f'{session_id}.pkl').exists():
                statistics['loaded_cached'] += 1
                assert not any('emb' in d for d in data_rec), data_rec
                assert not any('emb_samples' in d for d in data_rec), data_rec
                data_rec = pb.io.load(
                    dump_dir / f'{session_id}.pkl', unsafe=True)
                assert all('emb' in d for d in data_rec), data_rec
                assert all('emb_samples' in d for d in data_rec), data_rec
            else:
                statistics['calculated'] += 1
                assert not any('emb' in d for d in data_rec), data_rec
                assert not any('emb_samples' in d for d in data_rec), data_rec
                for d in data_rec:
                    audio = pb.io.load(d[audio_path])
                    if 'emb' not in d:
                        d['emb'] = self.resnet(audio)
                    if 'emb_samples' not in d:
                        d['emb_samples'] = audio.shape[-1]
                    del audio
                assert all('emb' in d for d in data_rec), data_rec
                assert all('emb_samples' in d for d in data_rec), data_rec
                if self.dump:
                    pb.io.dump(data_rec, dump_dir / f'{session_id}.pkl', unsafe=True)

            if num_speakers is None:
                n_clusters = len({d['speaker'] for d in data_rec})
                print(f'Use num_speakers: {n_clusters} for {session_id}')
            else:
                n_clusters = num_speakers

            yield session_id, data_rec, n_clusters

        statistics = dlp_mpi.gather(statistics)
        if dlp_mpi.IS_MASTER:
            new = {k: 0 for k in statistics[0].keys()}
            for s in statistics:
                for k, v in s.items():
                    new[k] += v
            statistics = new
            if len(set(statistics.values()) - {0}) == 1:
                print(f'Embedding statistics: {statistics}')
            else:
                print('#' * 79)
                print(f'Irregular embedding statistics: {statistics}. The data might be inconsistent.')
                print('#' * 79)


def chime7_reest(
        emb,
        speaker_ids,
        drop=False,
):
    """
    Code from our
        [1] Multi-stage diarization refinement for the CHiME-7 DASR scenario
    publication.
    Note: In [1], the prototypes are calculated using the diarization
          before TS-VAD, while this code will use the embeddings from
          the enhanced segment. The results are similar, but not identical.

    >>> emb = np.array([[1, 0], [1, 0.1], [0.9, 0], [0, 1], [0.1, 1]])
    >>> speaker_ids = ['a', 'a', 'a', 'b', 'b']
    >>> chime7_reest(emb, speaker_ids)
    [0, 0, 0, 1, 1]
    """
    unique_ids = sorted(set(speaker_ids))
    assert None not in unique_ids, unique_ids
    speaker_ids = np.array(speaker_ids)

    prototypes = np.array([
        np.mean(emb[speaker_ids == id_], axis=0)
        for id_ in unique_ids
    ])

    def dist(a, b):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)

        return 1 - np.einsum('...d,...d->...', a, b)

    labels = []
    for s, e in zip(speaker_ids, emb):
        d = dist(prototypes, e)

        d = [
            # np.amin(e) - 0.05 if pos == speaker_id else np.amax(e)
            # np.mean(e) - 0.05 if pos == speaker_id else np.mean(e)
            np.amin(e) - 0.05 if pos == s else np.mean(e)
            # 0 if pos == speaker_id else 1
            #         min(np.amin(e), 0.3) if pos == speaker_id else np.amax(e)
            #         np.median(e)
            for pos, e in zip(unique_ids, d)
        ]

        if s != np.argmin(d):
            if drop is True:
                if min(d) > 0.6:
                    labels.append(None)
                    continue
            elif drop is False:
                pass
            else:
                raise ValueError(drop)

        s = np.argmin(d)
        labels.append(s)

    assert len(labels) == len(speaker_ids), (len(labels), len(speaker_ids))
    return labels


class SC:
    version = 0
    scale = None

    def get_new_labels(
            self,
            data_rec,
            n_clusters,
    ):
        from sklearn.cluster import SpectralClustering

        emb = np.array([d['emb'] for d in data_rec])
        emb_samples = np.array([d['emb_samples'] for d in data_rec])

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels='discretize',
            random_state=0,
            affinity='precomputed',
        ).fit(similarity(emb, emb_samples, scale=self.scale, version=self.version))
        labels = clustering.labels_
        return labels

    def format_out(self, json):
        json = Path(json)
        return f'{json.parent}/{json.stem}_SLR_SC.json'

    def __call__(
            self,
            *jsons,
            audio_path='audio_path',
            num_speakers=None,
    ):
        if dlp_mpi.IS_MASTER:
            print(f'jsons: {jsons}')

        prepare = Prepare()
        for json in jsons:
            all_data = dlp_mpi.collection.UnorderedList()
            for session_id, data_rec, n_clusters in prepare.json_to_per_reco(json, audio_path, num_speakers):
                labels = self.get_new_labels(data_rec, n_clusters)
                assert len(labels) == len(data_rec), (len(labels), len(data_rec))
                for d, l in zip(data_rec, labels):
                    if l is not None:
                        d['speaker'] = str(l)
                        del d['emb']
                        all_data.append(d)

            all_data = all_data.gather()

            if dlp_mpi.IS_MASTER:
                out = self.format_out(json)
                pb.io.dump_json(all_data, out, indent=2, sort_keys=False)
                print(f'Wrote {out}')


class SC_step(SC):
    version = 1

    def format_out(self, json):
        json = Path(super().format_out(json))
        return f'{json.parent}/{json.stem}_step{self.scale}.json'

    def __call__(
            self,
            *jsons,
            audio_path='audio_path',
            num_speakers=None,
            alpha=0.25,
    ):
        if alpha == 1:
            print(f'Warning: alpha is 1. This is equivalent to SC.')
        elif alpha > 1:
            print(f'Warning: alpha is {alpha}. This is a uncommon value.')
        self.scale = alpha
        return super().__call__(
            *jsons, audio_path=audio_path, num_speakers=num_speakers)


class SC_poly(SC):
    version = 3

    def format_out(self, json):
        json = Path(super().format_out(json))
        return f'{json.parent}/{json.stem}_poly{self.scale}.json'

    def __call__(
            self,
            *jsons,
            audio_path='audio_path',
            num_speakers=None,
            beta=4,
    ):
        if beta == 1:
            print(f'Warning: beta is 1. This is equivalent to SC.')
        elif beta < 1:
            print(f'Warning: beta is {beta}. This is a uncommon value.')
        self.scale = beta
        return super().__call__(
            *jsons, audio_path=audio_path, num_speakers=num_speakers)


class Kmeans(SC):

    def format_out(self, json):
        json = Path(json)
        return f'{json.parent}/{json.stem}_SLR_kmeans.json'

    def get_new_labels(
            self,
            data_rec,
            n_clusters,
    ):
        from sklearn.cluster import KMeans

        emb = np.array([d['emb'] / np.linalg.norm(d['emb']) for d in data_rec])

        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=0,
        ).fit(emb)
        labels = clustering.labels_
        return labels


class C7sticky(SC):
    def format_out(self, json):
        json = Path(json)
        return f'{json.parent}/{json.stem}_SLR_C7sticky.json'

    def get_new_labels(
            self,
            data_rec,
            n_clusters,
    ):
        emb = np.array([d['emb'] for d in data_rec])
        speaker_ids = [d['speaker'] for d in data_rec]

        labels = chime7_reest(emb, speaker_ids, self.drop)
        return labels

    def __call__(
            self,
            *jsons,
            audio_path='audio_path',
            num_speakers=None,
            drop=False,
    ):
        self.drop = drop
        return super().__call__(
            *jsons, audio_path=audio_path, num_speakers=num_speakers)


def cli():
    import fire
    if sys.argv[1] == 'sbatch':
        sbatch(sys.argv[2:])
    else:
        fire.Fire({
            'sc': SC(),
            'sc_step': SC_step(),
            'sc_poly': SC_poly(),
            'c7sticky': C7sticky(),
            'kmeans': Kmeans(),
        })


if __name__ == '__main__':
    cli()
