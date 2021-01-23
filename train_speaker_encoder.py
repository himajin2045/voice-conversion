from pathlib import Path
import logging
import random
import argparse
import time
import pickle

import torch
import numpy as np

import dataset
from speaker_encoder import SpeakerEncoder
import hparams as hp

random.seed(hp.seed)

parser = argparse.ArgumentParser(description='speaker encoder')
parser.add_argument('--train', help='train mode', action='store_true')
parser.add_argument('--inference', help='inference mode', action='store_true')
parser.add_argument('--trace', help='trace model', action='store_true')
parser.add_argument('--model', help='load model')
args = parser.parse_args()

logging.basicConfig(
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=hp.log_path,
                level=logging.INFO)

def evaluate(model, loader, loss_device):
    steps = 0
    losses = []
    eers = []

    while True:
        if (steps+1) > hp.total_evaluate_steps:
            break

        for batch in loader:
            if (steps+1) > hp.total_evaluate_steps:
                break

            n_speakers, n_utterances, freq_len, tempo_len = batch.shape
            data = batch.view(-1, freq_len, tempo_len)
            data = data.transpose(1, 2)
            model.eval()
            embeds = model(data.cuda())
            embeds = embeds.view(n_speakers, n_utterances, -1)
            loss, eer = model.loss(embeds.to(loss_device))
            losses.append(loss.detach().numpy())
            eers.append(eer)
            steps += 1

    mean_loss = np.mean(losses)
    mean_eer = np.mean(eers)
    logging.info(f'evaluate mean loss {mean_loss:.3f}, mean eer {mean_eer:.3f}')

def train():
    stcmds_ds = dataset.new_stcmds_dataset(root=hp.stcmds_data_root, mel_feature_root=hp.mel_feature_root)
    aishell_ds = dataset.new_aishell_dataset(root=hp.aishell_data_root, mel_feature_root=hp.mel_feature_root)
    aidatatang_ds = dataset.new_aidatatang_dataset(root=hp.aidatatang_data_root, mel_feature_root=hp.mel_feature_root)
    primewords_ds = dataset.new_primewords_dataset(root=hp.primewords_data_root, mel_feature_root=hp.mel_feature_root)
    # toy_ds = dataset.new_toy_dataset(root=hp.toy_data_root, mel_feature_root=hp.mel_feature_root)

    datasets = [stcmds_ds, aishell_ds, aidatatang_ds, primewords_ds]
    # datasets = [stcmds_ds]
    # datasets = [toy_ds]
    mds = dataset.MultiAudioDataset(datasets)
    random.shuffle(mds.speakers)
    train_speakers = mds.speakers[:-50]
    eval_speakers = mds.speakers[-50:]

    ds = dataset.SpeakerDataset(train_speakers,
                        utterances_per_speaker=hp.utterances_per_speaker,
                        seq_len=hp.seq_len)
    loader = torch.utils.data.DataLoader(ds,
                                        batch_size=hp.speakers_per_batch,
                                        shuffle=True,
                                        num_workers=4)

    eval_ds = dataset.SpeakerDataset(eval_speakers,
                        utterances_per_speaker=hp.utterances_per_speaker,
                        seq_len=hp.seq_len)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
                                        batch_size=hp.speakers_per_batch,
                                        shuffle=True,
                                        num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")
    net = SpeakerEncoder(device, loss_device, 3)
    
    opt = torch.optim.Adam(net.parameters(), lr=hp.speaker_lr)

    total_steps = 0

    ckpts = sorted(list(Path(hp.save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt = torch.load(latest_ckpt_path)
        if ckpt:
            logging.info(f'loading ckpt {latest_ckpt_path}')
            net.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            total_steps = ckpt['total_steps']

    while True:
        if total_steps >= hp.train_steps:
            break

        for batch in loader:
            if total_steps >= hp.train_steps:
                break

            for g in opt.param_groups:
                g['lr'] = hp.speaker_lr
    
            n_speakers, n_utterances, freq_len, tempo_len = batch.shape
            data = batch.view(-1, freq_len, tempo_len)
            data = data.transpose(1, 2)

            net.train()
            opt.zero_grad()

            embeds = net(data.cuda())
            embeds = embeds.view(n_speakers, n_utterances, -1)
            loss, eer = net.loss(embeds.to(loss_device))

            loss.backward()
            net.do_gradient_ops()
            opt.step()

            total_steps += 1

            if (total_steps+1) % hp.train_print_interval == 0:
                logging.info(f'step {total_steps+1} loss {loss:.3f}, eer {eer:.3f}')
            if (total_steps+1) % hp.evaluate_interval == 0:
                evaluate(net, eval_loader, loss_device)
            if (total_steps+1) % hp.save_interval == 0:
                if not Path(hp.save_dir).exists():
                    Path(hp.save_dir).mkdir()
                save_path = Path(hp.save_dir) / f'{total_steps+1:012d}.pt'
                logging.info(f'saving ckpt {save_path}')
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'total_steps': total_steps
                }, save_path)

                # remove old ckpts
                ckpts = sorted(list(Path(hp.save_dir).glob('*.pt')))
                if len(ckpts) > hp.max_ckpts:
                    for ckpt in ckpts[:-hp.max_ckpts]:
                        Path(ckpt).unlink()
                        logging.info(f'ckpt {ckpt} removed')

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")
    net = SpeakerEncoder(device, loss_device, 3)

    ckpts = sorted(list(Path(hp.save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt_path = latest_ckpt_path
        if args.model is not None:
            ckpt_path = args.model
        ckpt = torch.load(ckpt_path)
        if ckpt:
            logging.info(f'loading ckpt {ckpt_path}')
            net.load_state_dict(ckpt['model_state_dict'])
        else:
            raise Exception('ckpt', 'no ckpts found')
    else:
        raise Exception('ckpt', 'no ckpts found')

    net.eval()

    speakers_per_batch = 32
    utterances_per_speaker = 10

    stcmds_ds = dataset.new_stcmds_dataset(root=hp.stcmds_data_root, mel_feature_root=hp.mel_feature_root)
    datasets = [stcmds_ds]
    mds = dataset.MultiAudioDataset(datasets)
    speakers = mds.speakers[:100]

    ds = dataset.SpeakerDataset(speakers,
                        utterances_per_speaker=utterances_per_speaker,
                        seq_len=hp.seq_len)

    loader = torch.utils.data.DataLoader(ds,
                                        batch_size=speakers_per_batch,
                                        shuffle=True,
                                        num_workers=4)

    for batch in loader:
        n_speakers, n_utterances, freq_len, tempo_len = batch.shape
        data = batch.view(-1, freq_len, tempo_len) # nxfxt
        data = data.transpose(1, 2) #nxtxf
        embeds = net(data.cuda())
        embeds = embeds.view(n_speakers, n_utterances, -1)
        embeds = embeds.to(loss_device)
        sim_matrix = net.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker,
                                         speakers_per_batch))
        results = sim_matrix.detach().cpu().numpy()
        with open('speaker_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        break

    # data = torch.randn(8, 80, 100)
    # data = data.transpose(1, 2)
    # embed = net(data.cuda())
    # logging.info(f'embeding shape {embed.shape}')

def trace():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")
    net = SpeakerEncoder(loss_device, loss_device, 3)
    ckpts = sorted(list(Path(hp.save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt = torch.load(latest_ckpt_path)
        if ckpt:
            logging.info(f'loading ckpt {latest_ckpt_path}')
            net.load_state_dict(ckpt['model_state_dict'])
        else:
            raise Exception('ckpt', 'no ckpts found')
    else:
        raise Exception('ckpt', 'no ckpts found')

    net.eval()

    x = torch.ones(1, 256, 80)
    sm = torch.jit.trace(net, x)
    sm.save('speaker_script_model.pt')

if __name__ == '__main__':
    if args.train:
        train()
    elif args.inference:
        inference()
    elif args.trace:
        trace()
    else:
        print('nothing to do')
