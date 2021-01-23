from typing import List
from pathlib import Path
import pickle
import logging
from multiprocessing import Process, JoinableQueue
import time
import json
import os
import random

import librosa
import torch
import numpy as np

import dsp
import hparams as hp


class Utterance(object):
    def __init__(self, id: str = None, raw_file: Path = None, mel_file: Path = None):
        self.id = id
        self.raw_file = raw_file
        self.mel_file = mel_file

    def raw(self, sr=16000, augment=False):
        """Get the raw audio samples."""

        y, sr = librosa.load(self.raw_file, sr=sr)
        # y, _ = librosa.effects.trim(y)
        if y.size == 0:
            raise Exception('audio', 'empty audio')
        y = 0.95 * librosa.util.normalize(y)
        if augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            y = y * amplitude
        return y

    def melspectrogram(self, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
        """Get the melspectrogram features."""

        try:
            # Load the serialized object if there is one.
            if self.mel_file is not None and self.mel_file.exists() and os.path.getsize(self.mel_file) > 0:
                with self.mel_file.open(mode='rb') as f:
                    mel = pickle.load(f)
                    if mel is None:
                        raise ValueError('loaded empty mel data')
                    logging.debug(f'loaded mel shape({mel.shape}) feautre {self.id} from {self.mel_file}')
                    return mel

            # Otherwise generate a new one from raw audio.
            # logging.debug(f'generated mel feautre {self.id}')
            # S = librosa.feature.melspectrogram(y=self.raw(sr=sr),
            #                                       sr=sr,
            #                                       n_fft=n_fft,
            #                                       hop_length=hop_length,
            #                                       win_length=win_length,
            #                                       n_mels=n_mels)
            # D = librosa.stft(y=self.raw(sr=sr),
            #             n_fft=n_fft,
            #             hop_length=hop_length,
            #             win_length=win_length)
            # S = 20 * np.log10(np.maximum(1e-5, np.abs(D))) - 16
            # return 10 * np.log10(np.maximum(1e-5, S))
            logging.debug(f'generate melspectrogram {self.id} from raw')
            return dsp.melspectrogram(self.raw(sr=hp.sample_rate))
        except Exception:
            logging.debug(f'failed to load melspectrogram, raw file: {self.raw_file}, mel file: {self.mel_file}')
            raise

    def random_raw_segment(self, seq_len):
        """Return a audio segment randomly."""

        y = self.raw(augment=True)
        ylen = len(y)
        if ylen < seq_len:
            pad_left = (seq_len - ylen) // 2
            pad_right = seq_len - ylen - pad_left
            y = np.pad(y, ((pad_left, pad_right)), mode='reflect')
        elif ylen > seq_len:
            max_seq_start = ylen - seq_len
            seq_start = np.random.randint(0, max_seq_start)
            seq_end = seq_start + seq_len
            y = y[seq_start:seq_end]

        return y

    def random_mel_segment(self, seq_len):
        """Return a melspectrogram segment randomly."""

        mel = self.melspectrogram()
        freq_len, tempo_len = mel.shape
        if tempo_len < seq_len:
            pad_left = (seq_len - tempo_len) // 2
            pad_right = seq_len - tempo_len - pad_left
            mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode='reflect')
        elif tempo_len > seq_len:
            max_seq_start = tempo_len - seq_len
            seq_start = np.random.randint(0, max_seq_start)
            seq_end = seq_start + seq_len
            mel = mel[:, seq_start:seq_end]
        return mel

class Speaker(object):
    def __init__(self, id: str):
        self.id = id
        self.utterances = []

    def add_utterance(self, utterance: Utterance):
        """Add an utterance to this speaker."""

        self.utterances.append(utterance)

    def random_utterances(self, n):
        """Return n utterances randomly."""

        return [self.utterances[idx] for idx in np.random.randint(0, len(self.utterances), n)]

class AudioDataset(object):
    def __init__(self, id: str, speakers: List[Speaker] = []):
        self.id = id
        self.speakers = speakers

    def add_speaker(self, speaker: Speaker):
        """Add a speaker to this dataset."""

        self.speakers.append(speaker)

    def random_speakers(self, n):
        """Return n speakers randomly."""

        return [self.speakers[idx] for idx in np.random.randint(0, len(self.speakers), n)]

    def serialize_speaker(self, queue: JoinableQueue, counter_queue: JoinableQueue):
        while True:
            speaker, root, overwrite = queue.get()

            if not root.exists():
                root.mkdir(parents=True)

            dsdir = root / self.id
            if not dsdir.exists():
                dsdir.mkdir()

            spkdir = dsdir / speaker.id
            if not spkdir.exists():
                spkdir.mkdir()

            for uttrn_idx, uttrn in enumerate(speaker.utterances):
                uttrnpath = spkdir / (uttrn.id + '.pkl')
                is_overwrite = False
                is_empty = False
                if uttrnpath.exists():
                    if os.path.getsize(uttrnpath) == 0:
                        logging.debug(f'overrite empty file {uttrnpath}')
                    elif not overwrite:
                        logging.debug(f'{uttrnpath} already exists, skip')
                        counter_queue.put(1)
                        continue
                    is_overwrite = True
                try:
                    mel = uttrn.melspectrogram()
                    with uttrnpath.open(mode='wb') as f:
                        pickle.dump(mel, f)
                    if is_overwrite:
                        logging.debug(f'dump pickle object to {uttrnpath} ({uttrn_idx+1}/{len(speaker.utterances)}), overwrite')
                    else:
                        logging.debug(f'dump pickle object to {uttrnpath} ({uttrn_idx+1}/{len(speaker.utterances)})')
                except Exception as err:
                    logging.warning(f'failed to dump mel features for file {uttrnpath}: {err}')
                counter_queue.put(1)
            queue.task_done()

    def serialization_counter(self, total_count, queue: JoinableQueue):
        count = 0
        while True:
            start_time = time.time()
            done = queue.get()
            duration = time.time() - start_time
            count += 1
            logging.debug(f'serialization progress {count}/{total_count}, {int(duration*1000)}ms/item')
            queue.task_done()

    def serialize_mel_feature(self, root: Path, overwrite=False):
        """Serialize melspectrogram features for all utterances of all speakers to the disk."""

        num_processes = 8
        queue = JoinableQueue()
        counter_queue = JoinableQueue()
        processes = []
        for i in range(num_processes):
            p = Process(target=self.serialize_speaker, args=(queue, counter_queue))
            processes.append(p)
            p.start()
        total_count = sum([len(spk.utterances) for spk in self.speakers])
        counter_process = Process(target=self.serialization_counter, args=(total_count, counter_queue))
        counter_process.start()
        # add tasks to queue
        logging.debug(f'total {len(self.speakers)} speakers')
        for spk in self.speakers:
            queue.put((spk, root, overwrite)) 
        # wait for all task done
        queue.join() 
        counter_queue.join()
        for p in processes:
            p.terminate()
        counter_process.terminate()

        # if not root.exists():
        #     root.mkdir(parents=True)

        # dsdir = root / self.id
        # if not dsdir.exists():
        #     dsdir.mkdir()

        # for spk_idx, spk in enumerate(self.speakers):
        #     spkdir = dsdir / spk.id
        #     if not spkdir.exists():
        #         spkdir.mkdir()

        #     for uttrn_idx, uttrn in enumerate(spk.utterances):
        #         uttrnpath = spkdir / (uttrn.id + '.pkl')
        #         is_overwrite = False
        #         if uttrnpath.exists():
        #             if not overwrite:
        #                 logging.debug(f'{uttrnpath} already exists, skip')
        #                 continue
        #             is_overwrite = True
        #         try:
        #             mel = uttrn.melspectrogram()
        #             with uttrnpath.open(mode='wb') as f:
        #                 pickle.dump(mel, f)
        #             if is_overwrite:
        #                 logging.debug(f'dump pickle object to {uttrnpath} ({uttrn_idx+1}/{len(spk.utterances)}) ({spk_idx+1}/{len(self.speakers)}), overwrite')
        #             else:
        #                 logging.debug(f'dump pickle object to {uttrnpath} ({uttrn_idx+1}/{len(spk.utterances)}) ({spk_idx+1}/{len(self.speakers)})')
        #         except ValueError as err:
        #             logging.warning(f'failed to dump mel features for file {uttrnpath}: {err}')

class MultiAudioDataset(object):
    def __init__(self, datasets: List[AudioDataset]):
        self.id = ''
        self.speakers = []
        ids = []
        for ds in datasets:
            ids.append(ds.id)
            self.speakers.extend(ds.speakers)
        self.id = '+'.join(ids)

class SpeakerDataset(object):
    def __init__(self, speakers, utterances_per_speaker, seq_len):
        self.speakers = speakers
        n_speakers = len(self.speakers)
        n_utterances = sum([len(spk.utterances) for spk in self.speakers])
        logging.info(f'total {n_speakers} speakers, {n_utterances} utterances')
        self.utterances_per_speaker = utterances_per_speaker
        self.seq_len = seq_len

    def random_utterance_segment(self, speaker_idx, seq_len):
        """Must return an utterance segment as long as the speaker has at least
        one effective utterance."""

        while True:
            try:
                utterance = self.speakers[speaker_idx].random_utterances(1)[0]
                return utterance.random_mel_segment(seq_len)
            except Exception as err:
                logging.debug(f'failed to load utterances of speaker idx {speaker_idx}: {err}')
                continue

    def __getitem__(self, idx):
        """Return random segments of random utterances for the specified speaker."""
        seq_len = 0
        if isinstance(self.seq_len, int):
            seq_len = self.seq_len
        elif isinstance(self.seq_len, list):
            seq_len = self.seq_len[random.randint(0, len(self.seq_len)-1)]
        else:
            raise ValueError('seq_len must be int or int list')

        segments = [self.random_utterance_segment(idx, seq_len) for _ in range(self.utterances_per_speaker)]
        return torch.tensor(segments)

    def __len__(self):
        return len(self.speakers)

class VocoderDataset(object):
    def __init__(self, speakers, utterances_per_speaker, seq_len):
        self.speakers = speakers
        n_speakers = len(self.speakers)
        n_utterances = sum([len(spk.utterances) for spk in self.speakers])
        logging.info(f'total {n_speakers} speakers, {n_utterances} utterances')
        self.utterances_per_speaker = utterances_per_speaker
        self.seq_len = seq_len

    def random_utterance_segment(self, speaker_idx):
        """Must return an utterance segment as long as the speaker has at least
        one effective utterance."""

        while True:
            try:
                utterance = self.speakers[speaker_idx].random_utterances(1)[0]
                segment = utterance.random_raw_segment(self.seq_len)
                # mel = dsp.melspectrogram(segment)
                # return segment, mel
                return segment
            except Exception as err:
                logging.debug(f'failed to load utterances: {err}')
                continue

    def __getitem__(self, idx):
        """Return random segments of random utterances for the specified speaker."""

        # segments = []
        # mels = []
        # for i in range(self.utterances_per_speaker):
        #     segment, mel = self.random_utterance_segment(idx)
        #     segments.append(segment)
        #     mels.append(mel)

        # # (n_uttrn x n_mels x time), (n_uttrn x frames)
        # return torch.tensor(segments), torch.tensor(mels)

        segments = [self.random_utterance_segment(idx) for _ in range(self.utterances_per_speaker)]
        return torch.tensor(segments)

    def __len__(self):
        return len(self.speakers)

class AdaGanDataset(object):
    def __init__(self, speakers, seq_len):
        self.speakers = speakers
        n_speakers = len(self.speakers)
        n_utterances = sum([len(spk.utterances) for spk in self.speakers])
        logging.info(f'total {n_speakers} speakers, {n_utterances} utterances')
        self.seq_len = seq_len

    def random_utterance_segment(self, speaker_idx):
        while True:
            try:
                utterance = self.speakers[speaker_idx].random_utterances(1)[0]
                segment = utterance.random_mel_segment(self.seq_len)
                return segment
            except Exception as err:
                logging.debug(f'failed to load utterances: {err}')
                continue

    def __getitem__(self, idx):
        x1 = self.random_utterance_segment(idx)
        x2 = self.random_utterance_segment(idx)
        rand_idx = np.random.randint(0, len(self.speakers))
        while rand_idx == idx:
            rand_idx = np.random.randint(0, len(self.speakers))
        y1 = self.random_utterance_segment(rand_idx)
        y2 = self.random_utterance_segment(rand_idx)
        return torch.tensor(x1), torch.tensor(x2), torch.tensor(y1), torch.tensor(y2)

    def __len__(self):
        return len(self.speakers)

def new_stcmds_dataset(root: Path, mel_feature_root: Path):
    """Load the stcmds dataset into an AudioDataset.

    855 speakers

    About the dataset: http://openslr.org/38/
    """

    dataset_id = 'stcmds'
    id2speaker = dict()
    wav_files = root.glob('*.wav')
    for f in wav_files:
        speaker_id = f.name[8:-7]
        utterance_id = f.name[-7:-4]
        if mel_feature_root is not None:
            mel_f = mel_feature_root / dataset_id / speaker_id / (utterance_id + '.pkl')
        uttrn = Utterance(utterance_id, raw_file=f, mel_file=mel_f)
        if speaker_id in id2speaker:
            id2speaker[speaker_id].add_utterance(uttrn)
        else:
            spk = Speaker(speaker_id)
            spk.add_utterance(uttrn)
            id2speaker[speaker_id] = spk

    dataset = AudioDataset(dataset_id, speakers=list(id2speaker.values()))
    return dataset

def new_aishell_dataset(root: Path, mel_feature_root: Path):
    """Load the aishell dataset into an AudioDataset.

    The root folder should contains uncompressed train/dev/test folders each
    contains multiple speaker folders where each speaker folder contains
    multiple wav files.

    400 speakers

    About the aishell dataset: http://openslr.org/33/
    """

    dataset_id = 'aishell'
    id2speaker = dict()
    wav_files = root.rglob('*.wav')
    for f in wav_files:
        speaker_id = f.parent.name # use the parent folder name as speaker id
        utterance_id = f.name[:-4] # use the file name as the utterance id
        if mel_feature_root is not None:
            mel_f = mel_feature_root / dataset_id / speaker_id / (utterance_id + '.pkl')
        uttrn = Utterance(utterance_id, raw_file=f, mel_file=mel_f)
        if speaker_id in id2speaker:
            id2speaker[speaker_id].add_utterance(uttrn)
        else:
            spk = Speaker(speaker_id)
            spk.add_utterance(uttrn)
            id2speaker[speaker_id] = spk

    dataset = AudioDataset(dataset_id, speakers=list(id2speaker.values()))
    return dataset

def new_aidatatang_dataset(root: Path, mel_feature_root: Path):
    """Load the aidatatang_200zh dataset into an AudioDataset.

    The root folder should contains uncompressed train/dev/test folders each
    contains multiple speaker folders where each speaker folder contains
    multiple wav files.

    600 speakers

    About the aishell dataset: http://openslr.org/62/
    """

    dataset_id = 'aidatatang_200zh'
    id2speaker = dict()
    wav_files = root.rglob('*.wav')
    for f in wav_files:
        speaker_id = f.parent.name # use the parent folder name as speaker id
        utterance_id = f.name[:-4] # use the file name as the utterance id
        if mel_feature_root is not None:
            mel_f = mel_feature_root / dataset_id / speaker_id / (utterance_id + '.pkl')
        uttrn = Utterance(utterance_id, raw_file=f, mel_file=mel_f)
        if speaker_id in id2speaker:
            id2speaker[speaker_id].add_utterance(uttrn)
        else:
            spk = Speaker(speaker_id)
            spk.add_utterance(uttrn)
            id2speaker[speaker_id] = spk

    dataset = AudioDataset(dataset_id, speakers=list(id2speaker.values()))
    return dataset

def new_libritts_dataset(root: Path, mel_feature_root: Path):
    """Load the libritts dataset into an AudioDataset.

    About the dataset: http://openslr.org/60/
    """

    dataset_id = 'libritts'
    id2speaker = dict()
    wav_files = root.rglob('*.wav')
    for f in wav_files:
        utterance_id = f.name[:-4]
        speaker_id = utterance_id.split('_')[0]
        if utterance_id == '' or speaker_id == '':
            logging.warning(f'{dataset_id} can not extract speaker_id/utterance_id from file {f}')
            continue
        if mel_feature_root is not None:
            mel_f = mel_feature_root / dataset_id / speaker_id / (utterance_id + '.pkl')
        uttrn = Utterance(utterance_id, raw_file=f, mel_file=mel_f)
        if speaker_id in id2speaker:
            id2speaker[speaker_id].add_utterance(uttrn)
        else:
            spk = Speaker(speaker_id)
            spk.add_utterance(uttrn)
            id2speaker[speaker_id] = spk

    dataset = AudioDataset(dataset_id, speakers=list(id2speaker.values()))
    return dataset

def new_primewords_dataset(root: Path, mel_feature_root: Path):
    """Load the primewords dataset into an AudioDataset.

    296 speakers

    About the dataset: http://openslr.org/47/
    """

    def find_uttrn_by_fn(trans, fn):
        for uttrn in trans:
            if uttrn['file'] == fn:
                return uttrn
        return None

    dataset_id = 'primewords'
    trans_file = root / 'set1_transcript.json'
    with trans_file.open() as f:
        trans = json.load(f)

    id2speaker = dict()
    wav_files = root.rglob('*.wav')
    for f in wav_files:
        uttrn = find_uttrn_by_fn(trans, f.name)
        if uttrn is None:
            logging.warning(f'{dataset_id} can not extract speaker_id/utterance_id from file {f}')
            continue
        utterance_id = uttrn['id'] + '_' + f.name[:-4]
        speaker_id = uttrn['user_id']
        if utterance_id == '' or speaker_id == '':
            logging.warning(f'{dataset_id} can not extract speaker_id/utterance_id from file {f}')
            continue
        if mel_feature_root is not None:
            mel_f = mel_feature_root / dataset_id / speaker_id / (utterance_id + '.pkl')
        uttrn = Utterance(utterance_id, raw_file=f, mel_file=mel_f)
        if speaker_id in id2speaker:
            id2speaker[speaker_id].add_utterance(uttrn)
        else:
            spk = Speaker(speaker_id)
            spk.add_utterance(uttrn)
            id2speaker[speaker_id] = spk

    dataset = AudioDataset(dataset_id, speakers=list(id2speaker.values()))
    return dataset

def new_toy_dataset(root: Path, mel_feature_root: Path):
    """Load the toy dataset spliting from the aishell dataset into an AudioDataset.

    8 speakers
    """

    dataset_id = 'toy'
    id2speaker = dict()
    wav_files = root.rglob('*.wav')
    for f in wav_files:
        speaker_id = f.parent.name # use the parent folder name as speaker id
        utterance_id = f.name[:-4] # use the file name as the utterance id
        if mel_feature_root is not None:
            mel_f = mel_feature_root / dataset_id / speaker_id / (utterance_id + '.pkl')
        uttrn = Utterance(utterance_id, raw_file=f, mel_file=mel_f)
        if speaker_id in id2speaker:
            id2speaker[speaker_id].add_utterance(uttrn)
        else:
            spk = Speaker(speaker_id)
            spk.add_utterance(uttrn)
            id2speaker[speaker_id] = spk

    dataset = AudioDataset(dataset_id, speakers=list(id2speaker.values()))
    return dataset
