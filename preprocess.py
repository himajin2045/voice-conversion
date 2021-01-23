from pathlib import Path
import logging

import dataset
import hparams as hp

logging.basicConfig(filename='/tmp/preprocess.log', level=logging.DEBUG)

# mel_feature_root = Path('/mnt/ssd500/speech/autovc/data')
mel_feature_root = Path(hp.mel_feature_root)


stcmds_ds = dataset.new_stcmds_dataset(root=hp.stcmds_data_root, mel_feature_root=hp.mel_feature_root)
stcmds_ds.serialize_mel_feature(hp.mel_feature_root)

aishell_ds = dataset.new_aishell_dataset(root=hp.aishell_data_root, mel_feature_root=hp.mel_feature_root)
aishell_ds.serialize_mel_feature(hp.mel_feature_root)

aidatatang_ds = dataset.new_aidatatang_dataset(root=hp.aidatatang_data_root, mel_feature_root=hp.mel_feature_root)
aidatatang_ds.serialize_mel_feature(hp.mel_feature_root)

primewords_ds = dataset.new_primewords_dataset(root=hp.primewords_data_root, mel_feature_root=hp.mel_feature_root)
primewords_ds.serialize_mel_feature(hp.mel_feature_root)

# toy_data_root = Path('/mnt/ssd500/dataset/speech/toy')
# ds = dataset.new_toy_dataset(root=toy_data_root, mel_feature_root=mel_feature_root)
# ds.serialize_mel_feature(mel_feature_root)

# load/generate mel features
# for spk in ds.speakers:
#     for uttrn in spk.utterances:
#         mel = uttrn.melspectrogram()
#         logging.info(f'mel feature for speaker {spk.id} utterance {uttrn.id}, shape {mel.shape}')
