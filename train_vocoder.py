import torch
import random
import argparse
import logging
from pathlib import Path
import pickle

import torch.nn.functional as F

from model_vocoder import Generator
from model_vocoder import Discriminator
from model_vocoder import Audio2Mel
import hparams as hp
import dataset
import dsp

random.seed(hp.seed)

parser = argparse.ArgumentParser(description='generator')
parser.add_argument('--train', help='train mode', action='store_true')
parser.add_argument('--inference', help='inference mode', action='store_true')
parser.add_argument('--model', help='load model')
parser.add_argument('--trace', help='trace', action='store_true')
args = parser.parse_args()

logging.basicConfig(
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=hp.log_path,
                level=logging.INFO)

def train():
    # v1, all ds
    # v3, stcmds ds alone

    stcmds_ds = dataset.new_stcmds_dataset(root=hp.stcmds_data_root, mel_feature_root=hp.mel_feature_root)
    # aishell_ds = dataset.new_aishell_dataset(root=hp.aishell_data_root, mel_feature_root=hp.mel_feature_root)
    # aidatatang_ds = dataset.new_aidatatang_dataset(root=hp.aidatatang_data_root, mel_feature_root=hp.mel_feature_root)
    # primewords_ds = dataset.new_primewords_dataset(root=hp.primewords_data_root, mel_feature_root=hp.mel_feature_root)
    # datasets = [stcmds_ds, aishell_ds, aidatatang_ds, primewords_ds]
    datasets = [stcmds_ds]
    mds = dataset.MultiAudioDataset(datasets)
    random.shuffle(mds.speakers)
    train_speakers = mds.speakers
    # eval_speakers = mds.speakers[-100:]
    
    ds = dataset.VocoderDataset(train_speakers,
                        utterances_per_speaker=1,
                        seq_len=hp.vocoder_seq_len)
    loader = torch.utils.data.DataLoader(ds,
                                        batch_size=hp.vocoder_batch_size,
                                        shuffle=True,
                                        num_workers=6)

    netG = Generator(hp.num_mels, hp.vocoder_ngf, hp.vocoder_n_residual_layers).cuda()
    netD = Discriminator(hp.vocoder_num_D, hp.vocoder_ndf, hp.vocoder_n_layers_D, hp.vocoder_downsamp_factor).cuda()
    fft = Audio2Mel(n_fft=hp.n_fft,
                    hop_length=hp.hop_length,
                    win_length=hp.win_length,
                    sampling_rate=hp.sample_rate,
                    n_mel_channels=hp.num_mels,
                    mel_fmin=hp.fmin,
                    mel_fmax=hp.fmax,
                    min_level_db=hp.min_level_db).cuda()

    optG = torch.optim.Adam(netG.parameters(), lr=hp.vocoder_G_lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=hp.vocoder_D_lr, betas=(0.5, 0.9))

    total_steps = 0

    ckpts = sorted(list(Path(hp.vocoder_save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt = torch.load(latest_ckpt_path)
        if ckpt:
            logging.info(f'loading vocoder ckpt {latest_ckpt_path}')
            netG.load_state_dict(ckpt['netG_state_dict'])
            netD.load_state_dict(ckpt['netD_state_dict'])
            optG.load_state_dict(ckpt['optG_state_dict'])
            optD.load_state_dict(ckpt['optD_state_dict'])
            total_steps = ckpt['total_steps']

    while True:
        if total_steps >= hp.vocoder_train_steps:
            break

        for segments in loader:
            if total_steps >= hp.vocoder_train_steps:
                break

            x_t = segments.cuda()
            s_t = fft(x_t).detach()
            # print(f's_t.shape {s_t.shape}')
            x_pred_t = netG(s_t.cuda())
            # print(f'x_pred_t {x_pred_t.shape}')

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            #######################
            # Train Discriminator #
            #######################
            D_fake_det = netD(x_pred_t.cuda().detach())
            D_real = netD(x_t.cuda())

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.cuda())

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (hp.vocoder_n_layers_D + 1) # 0.8
            D_weights = 1.0 / hp.vocoder_num_D # 0.33333
            wt = D_weights * feat_weights # 2.666666
            for i in range(hp.vocoder_num_D):
                for j in range(len(D_fake[i]) - 1):
                    print(f'i,j {i},{j} {D_fake[i][j]}')
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + 10. * loss_feat).backward()
            optG.step()

            total_steps += 1

            if (total_steps+1) % hp.vocoder_train_print_interval == 0:
                logging.info(f'vocoder step {total_steps+1} loss discriminator {loss_D.item():.3f} generator {loss_G.item():.3f} FM {loss_feat.item():.3f} recon {s_error:.3f}')
            if (total_steps+1) % hp.vocoder_save_interval == 0:
                if not Path(hp.vocoder_save_dir).exists():
                    Path(hp.vocoder_save_dir).mkdir()
                save_path = Path(hp.vocoder_save_dir) / f'{total_steps+1:012d}.pt'
                logging.info(f'saving vocoder ckpt {save_path}')
                torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optG_state_dict': optG.state_dict(),
                    'optD_state_dict': optD.state_dict(),
                    'total_steps': total_steps
                }, save_path)

                # remove old ckpts
                ckpts = sorted(list(Path(hp.vocoder_save_dir).glob('*.pt')))
                if len(ckpts) > hp.vocoder_max_ckpts:
                    for ckpt in ckpts[:-hp.vocoder_max_ckpts]:
                        Path(ckpt).unlink()
                        logging.info(f'ckpt {ckpt} removed')

def inference():
    netG = Generator(hp.num_mels, hp.vocoder_ngf, hp.vocoder_n_residual_layers).cuda()
    fft = Audio2Mel(n_fft=hp.n_fft,
                    hop_length=hp.hop_length,
                    win_length=hp.win_length,
                    sampling_rate=hp.sample_rate,
                    n_mel_channels=hp.num_mels,
                    mel_fmin=hp.fmin,
                    mel_fmax=hp.fmax,
                    min_level_db=hp.min_level_db).cuda()
    if args.model:
        logging.info(f'loading vocoder ckpt {args.model}')
        ckpt = torch.load(args.model)
    else:
        ckpts = sorted(list(Path(hp.vocoder_save_dir).glob('*.pt')))
        if len(ckpts) > 0:
            latest_ckpt_path = ckpts[-1]
            logging.info(f'loading vocoder ckpt {latest_ckpt_path}')
            ckpt = torch.load(latest_ckpt_path)
    if ckpt:
        netG.load_state_dict(ckpt['netG_state_dict'])
    else:
        print('no checkpoints')

    # seen sample
    # f = '/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00442A0027.wav'
    # f = '/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00435A0028.wav'

    # english (unseen speaker)
    # f = '/mnt/ssd500/dataset/speech/libritts/extract/LibriTTS/train-other-500/428/125877/428_125877_000015_000000.wav'
    # f = '/mnt/ssd500/dataset/speech/libritts/extract/LibriTTS/train-other-500/428/125877/428_125877_000087_000001.wav'
    # f = '/mnt/ssd500/dataset/speech/libritts/extract/LibriTTS/train-other-500/428/125877/428_125877_000065_000000.wav'

    # chinese (unseen speaker)
    f = '/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00213I0037.wav'
    # f = '/mnt/wd500/dataset/speech/cn-celeb/extract/CN-Celeb/eval/enroll/id00987-enroll.wav'
    # f = '/mnt/wd500/dataset/speech/cn-celeb/extract/CN-Celeb/eval/enroll/id00998-enroll.wav'
    # f = '/mnt/wd500/dataset/speech/cn-celeb/extract/CN-Celeb/eval/enroll/id00960-enroll.wav'
    # f = '/mnt/wd500/dataset/speech/MAGICDATA-SLR68/extract/train/14_4030/14_4030_20170905174343.wav'

    # chinese (seen speaker)
    # f = '/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00014A0120.wav'
    # f = '/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00096I0120.wav'
    # f = '/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00122A0120.wav'
    # f = '/mnt/ssd500/dataset/speech/aishell/data_aishell/wav/test/S0902/BAC009S0902W0477.wav'
    # f = '/mnt/ssd500/dataset/speech/primewords/extract/primewords_md_2018_set1/audio_files/0/0e/0eb1f442-f6b3-4e8c-abd7-e5720b4bdb99.wav'
    # f = '/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00179A0076.wav'
    # f = '/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00001A0003.wav'
    # f = '/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00001A0027.wav'
    # f = '/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00299I0067.wav'
    # f = '/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00299I0070.wav'
    # f = '/mnt/ssd500/dataset/speech/aishell/data_aishell/wav/train/S0169/BAC009S0169W0317.wav'
    # f = '/mnt/ssd500/dataset/speech/aishell/data_aishell/wav/train/S0169/BAC009S0169W0400.wav'
    uttrn = dataset.Utterance(id=None, raw_file=f)
    y = torch.from_numpy(uttrn.raw(sr=hp.sample_rate)).cuda()
    S = fft(y.unsqueeze(0).unsqueeze(0))
    # S = torch.from_numpy(uttrn.melspectrogram()).cuda().unsqueeze(0)
    y_pred = netG(S)
    S_recon = fft(y_pred)
    l1loss = F.l1_loss(S, S_recon)
    mseloss = F.mse_loss(S, S_recon)
    print(f'y.shape {y.shape}, S.shape {S.shape} y_pred.shape {y_pred.shape}')
    print(f'S.mean {S.mean()} S_recon.mean {S_recon.mean()} l1loss {l1loss:.5f} mseloss {mseloss:.5f}')
    results = [y.detach().cpu().numpy(), S.detach().cpu().numpy(), y_pred.detach().cpu().numpy()]
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

def trace():
    netG = Generator(hp.num_mels, hp.vocoder_ngf, hp.vocoder_n_residual_layers)
    if args.model:
        logging.info(f'loading vocoder ckpt {args.model}')
        ckpt = torch.load(args.model)
    else:
        ckpts = sorted(list(Path(hp.vocoder_save_dir).glob('*.pt')))
        if len(ckpts) > 0:
            latest_ckpt_path = ckpts[-1]
            logging.info(f'loading vocoder ckpt {latest_ckpt_path}')
            print(f'loading vocoder ckpt {latest_ckpt_path}')
            ckpt = torch.load(latest_ckpt_path)
    if ckpt:
        netG.load_state_dict(ckpt['netG_state_dict'])
    else:
        print('no checkpoints')

    x = torch.ones(1, 80, 298)
    sm = torch.jit.trace(netG, x)
    print(sm.code)
    y = sm(x)
    sm.save('vocoder_script_model.pt')
    print(y.shape)

    # fft = Audio2Mel(n_fft=hp.n_fft,
    #                 hop_length=hp.hop_length,
    #                 win_length=hp.win_length,
    #                 sampling_rate=hp.sample_rate,
    #                 n_mel_channels=hp.num_mels,
    #                 mel_fmin=hp.fmin,
    #                 mel_fmax=hp.fmax,
    #                 min_level_db=hp.min_level_db)
    # f = '/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00213I0037.wav'
    # uttrn = dataset.Utterance(id=None, raw_file=f)
    # y = torch.from_numpy(uttrn.raw(sr=hp.sample_rate))
    # y = y.unsqueeze(0).unsqueeze(0)
    # print(f'fft input {y.shape}') # (1, 1, 66704)
    # sm = torch.jit.trace(fft, y)
    # mel = sm(y)
    # print(f'mel {mel.shape}')
    # print(mel)
    # # sm.save('fft_script_model.pt')
    # print(sm.code)

if __name__ == '__main__':
    if args.train:
        train()
    elif args.inference:
        inference()
    elif args.trace:
        trace()
    else:
        print('nothing to do')
