from pathlib import Path
import logging
import random
import pickle
import math
import argparse
import shutil

import torch
import numpy as np
import torch.nn.functional as F

import dataset
import hparams as hp
from model_vc import Generator, Encoder, Decoder, Postnet
from speaker_encoder import SpeakerEncoder
import model_vocoder

random.seed(hp.seed)

parser = argparse.ArgumentParser(description='generator')
parser.add_argument('--train', help='train mode', action='store_true')
parser.add_argument('--inference', help='inference mode', action='store_true')
parser.add_argument('--trace', help='trace', action='store_true')
parser.add_argument('--pretrained', help='pretrained model')
parser.add_argument('--model', help='pretrained model')
args = parser.parse_args()

logging.basicConfig(
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=hp.log_path,
                level=logging.INFO)

def train():
    stcmds_ds = dataset.new_stcmds_dataset(root=hp.stcmds_data_root, mel_feature_root=hp.mel_feature_root)
    # aishell_ds = dataset.new_aishell_dataset(root=hp.aishell_data_root, mel_feature_root=hp.mel_feature_root)
    # aidatatang_ds = dataset.new_aidatatang_dataset(root=hp.aidatatang_data_root, mel_feature_root=hp.mel_feature_root)
    # primewords_ds = dataset.new_primewords_dataset(root=hp.primewords_data_root, mel_feature_root=hp.mel_feature_root)
    # toy_ds = dataset.new_toy_dataset(root=hp.toy_data_root, mel_feature_root=hp.mel_feature_root)

    # datasets = [stcmds_ds, aishell_ds, aidatatang_ds, primewords_ds]
    datasets = [stcmds_ds]
    # datasets = [toy_ds]
    mds = dataset.MultiAudioDataset(datasets)
    random.shuffle(mds.speakers)
    train_speakers = mds.speakers[:-40]
    eval_speakers = mds.speakers[-40:]

    ds = dataset.SpeakerDataset(train_speakers,
                        utterances_per_speaker=hp.generator_utterances_per_speaker,
                        seq_len=hp.generator_seq_len)
    loader = torch.utils.data.DataLoader(ds,
                                        batch_size=hp.generator_speakers_per_batch,
                                        shuffle=True,
                                        num_workers=6)

    eval_ds = dataset.SpeakerDataset(eval_speakers,
                        utterances_per_speaker=hp.generator_utterances_per_speaker,
                        seq_len=hp.generator_seq_len)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
                                        batch_size=hp.generator_speakers_per_batch,
                                        shuffle=True,
                                        num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    speaker_encoder = SpeakerEncoder(device, loss_device, 3)
    ckpts = sorted(list(Path(hp.save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt = torch.load(latest_ckpt_path)
        if ckpt:
            logging.info(f'loading speaker encoder ckpt {latest_ckpt_path}')
            speaker_encoder.load_state_dict(ckpt['model_state_dict'])
        else:
            raise Exception('ckpt', 'no ckpts found')
    else:
        raise Exception('ckpt', 'no ckpts found')
    speaker_encoder.eval()

    # generator = Generator(32, 256, 512, 16) # train_speakers[:120] g5_ckpts_bak
    # generator = Generator(32, 256, 512, 16) # train_speakers[:800] g6_ckpts_bak
    # generator = Generator(32, 256, 512, 16) # train_speakers[:800] g12_ckpts_bak 3layers-speaker_encoder
    # generator = Generator(16, 256, 512, 16) # train_speakers[:800] g13_ckpts_bak 3layers-speaker_encoder
    # generator = Generator(24, 256, 512, 16) # train_speakers[:800] g14_ckpts_bak 3layers-speaker_encoder
    # generator = Generator(24, 256, 512, 16) # [stcmds_ds, aishell_ds, aidatatang_ds, primewords_ds] g15_ckpts_bak 3layers-speaker_encoder
    # use src emb from a different utterance
    # use variate seq_len (128, 256, ...)
    # generator = Generator(24, 256, 512, 16) # train_speakers[:800] g16_ckpts_bak 3layers-speaker_encoder var-seqlen (128train->256finetune) diff-emb
    # generator = Generator(8, 256, 512, 4) # train_speakers[:800] g17_ckpts_bak 3layers-speaker_encoder
    generator = Generator(8, 256, 512, 4) # train_speakers[:800] g18_ckpts_bak 3layers-speaker_encoder bs-16
    # large batch size
    # speaker code reconstruct
    # generator = Generator(32, 256, 512, 8) train_speakers[:120] g7
    # generator = Generator(32, 256, 512, 8) # train_speakers[:800] g11
    # generator = Generator(32, 256, 512, 2) [:120] g8
    # generator = Generator(32, 256, 512, 2) [:800] g9
    # generator = Generator(16, 256, 512, 2) [:800] # g10
    # generator = Generator(16, 256, 512, 2)
    generator.to(device=device)

    opt = torch.optim.Adam(generator.parameters(), lr=hp.generator_lr)
    total_steps = 0

    ckpts = sorted(list(Path(hp.generator_save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt = torch.load(latest_ckpt_path)
        if ckpt:
            logging.info(f'loading generator ckpt {latest_ckpt_path}')
            generator.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            total_steps = ckpt['total_steps']

    if args.pretrained:
        ckpt = torch.load(args.pretrained)
        generator.load_state_dict(ckpt['model_state_dict'])
        logging.info(f'loaded pretrained model {args.pretrained}')

    while True:
        if total_steps >= hp.generator_train_steps:
            break

        for batch in loader:
            if total_steps >= hp.generator_train_steps:
                break

            for param_group in opt.param_groups:
                param_group['lr'] = hp.generator_get_lr(total_steps+1)

            generator.train()

            batch = batch.cuda()
            n_speakers, n_utterances, freq_len, tempo_len = batch.shape
            data = batch.view(-1, freq_len, tempo_len)
            embeds = speaker_encoder(data.transpose(1, 2)).detach()
            embeds = embeds.view(n_speakers, n_utterances, -1)

            # assert batch.size(1) == 2
            src_mels = batch[:, 0, :, :]
            src_mels = src_mels.transpose(1, 2)
            # logging.info(f'src_mels.shape {src_mels.shape}')

            # assert embeds.size(1) == 2
            # src_embeds = embeds.mean(dim=1) # average the embeddings
            # Target embed from the same speaker as source embed in training phase,
            # and should be a different speaker in inference phase. Here the target
            # utterance is also different from the source utterance.
            src_embeds = embeds[:, 0, :] 
            # logging.info(f'embeds.shape {src_embeds.shape} {tgt_embeds.shape}')

            init_out, final_out, content_out, code_exp = generator(src_mels, src_embeds, src_embeds.unsqueeze(1))
            # content_out2 = generator(batch[:, 1, :, :].transpose(1, 2), tgt_embeds, None)
            # logging.info(f'out shapes {init_out.shape} {final_out.shape} {content_out.shape}')

            # content_diff_loss = F.cosine_similarity(content_out.view(1, -1), content_out2.view(1, -1)).mean()

            loss, recon_loss, recon0_loss, content_recon_loss = generator.loss(src_mels,
                            src_embeds,
                            init_out,
                            final_out,
                            content_out)

            opt.zero_grad()
            # (loss + 0.3 * content_diff_loss).backward()
            loss.backward()
            opt.step()
            total_steps += 1

            if (total_steps+1) % hp.generator_train_print_interval == 0:
                logging.info(f'generator step {total_steps+1} loss {loss:.3f} ==> recon_loss {recon_loss:.3f} recon0_loss {recon0_loss:.3f} content_recon_loss {content_recon_loss:.5f}')
            if (total_steps+1) % hp.generator_evaluate_interval == 0:
                evaluate(generator, speaker_encoder, eval_loader)
            if (total_steps+1) % hp.generator_save_interval == 0:
                if not Path(hp.generator_save_dir).exists():
                    Path(hp.generator_save_dir).mkdir()
                save_path = Path(hp.generator_save_dir) / f'{total_steps+1:012d}.pt'
                logging.info(f'saving generrator ckpt {save_path}')
                torch.save({
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'total_steps': total_steps
                }, save_path)

                # remove old ckpts
                ckpts = sorted(list(Path(hp.generator_save_dir).glob('*.pt')))
                if len(ckpts) > hp.generator_max_ckpts:
                    for ckpt in ckpts[:-hp.generator_max_ckpts]:
                        Path(ckpt).unlink()
                        logging.info(f'ckpt {ckpt} removed')
            # if (total_steps+1) % hp.generator_bak_interval == 0:
            #     if not Path(hp.generator_bak_dir).exists():
            #         Path(hp.generator_bak_dir).mkdir()
            #     ckpts = sorted(list(Path(hp.generator_save_dir).glob('*.pt')))
            #     shutil.copy(ckpts[-1], hp.generator_bak_dir)
            #     logging.info(f'ckpt {ckpts[-1]} backuped')
            if (total_steps+1) % hp.generator_sample_interval == 0:
                results = [
                    src_mels.detach().cpu().numpy(),
                    final_out.detach().cpu().numpy(),
                    content_out.detach().cpu().numpy(),
                    code_exp.detach().cpu().numpy(),
                ]
                with open('generator_samples.pkl', 'wb') as f:
                    pickle.dump(results, f)
                pass

def evaluate(generator, speaker_encoder, loader):
    steps = 0
    losses = []

    while True:
        if (steps+1) > hp.total_evaluate_steps:
            break

        for batch in loader:
            if (steps+1) > hp.total_evaluate_steps:
                break

            batch = batch.cuda()
            n_speakers, n_utterances, freq_len, tempo_len = batch.shape
            data = batch.view(-1, freq_len, tempo_len)
            data = data.transpose(1, 2)
            embeds = speaker_encoder(data).detach()
            embeds = embeds.view(n_speakers, n_utterances, -1)

            # assert batch.size(1) == 2
            src_mels = batch[:, 0, :, :]
            src_mels = src_mels.transpose(1, 2)
            # logging.info(f'src_mels.shape {src_mels.shape}')

            # assert embeds.size(1) == 2
            src_embeds = embeds[:, 0, :]
            # Target embed from the same speaker as source embed in training phase,
            # and should be a different speaker in inference phase. Here the target
            # utterance is also different from the source utterance.
            # tgt_embeds = embeds[:, 1, :] 
            # logging.info(f'embeds.shape {src_embeds.shape} {tgt_embeds.shape}')

            generator.eval()
            init_out, final_out, content_out, _ = generator(src_mels, src_embeds, src_embeds.unsqueeze(1))
            # logging.info(f'out shapes {init_out.shape} {final_out.shape} {content_out.shape}')

            loss, recon_loss, recon0_loss, content_recon_loss = generator.loss(src_mels,
                            src_embeds,
                            init_out,
                            final_out,
                            content_out)
            losses.append(loss.detach().cpu().numpy())
            steps += 1

    mean_loss = np.mean(losses)
    logging.info(f'generator evaluate mean loss {mean_loss:.3f}')

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    speaker_encoder = SpeakerEncoder(device, loss_device, 3)
    ckpts = sorted(list(Path(hp.save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt = torch.load(latest_ckpt_path)
        if ckpt:
            logging.info(f'loading speaker encoder ckpt {latest_ckpt_path}')
            speaker_encoder.load_state_dict(ckpt['model_state_dict'])
        else:
            raise Exception('ckpt', 'no ckpts found')
    else:
        raise Exception('ckpt', 'no ckpts found')

    generator = Generator(8, 256, 512, 4)
    ckpts = sorted(list(Path(hp.generator_save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        ckpt = torch.load(latest_ckpt_path)
        if ckpt:
            logging.info(f'loading generator ckpt {latest_ckpt_path}')
            generator.load_state_dict(ckpt['model_state_dict'])

    generator.to(device=device)
    speaker_encoder.eval()
    generator.eval()

    # pad with zeros to the end of the time axis
    def pad_zeros(x):
        mul = math.ceil(float(x.shape[1]) / 32)
        pad_len = mul*32 - x.shape[1]
        return np.pad(x, pad_width=((0, 0), (0, pad_len)), mode='constant')

    def pad_zeros_multi(xs):
        max_len = 0
        for x in xs:
            if x.shape[1] > max_len:
                max_len = x.shape[1]

        newxs = []
        for x in xs:
            mul = math.ceil(float(max_len) / 32)
            pad_len = mul*32 - x.shape[1]
            newxs.append(np.pad(x, pad_width=((0, 0), (0, pad_len)), mode='constant'))

        return newxs


    stcmds_ds = dataset.new_stcmds_dataset(root=hp.stcmds_data_root, mel_feature_root=hp.mel_feature_root)
    datasets = [stcmds_ds]
    mds = dataset.MultiAudioDataset(datasets)
    random.shuffle(mds.speakers)
    speakers = mds.speakers

    # src_uttrn = speakers[1].random_utterances(1)[0]
    src_uttrn = dataset.Utterance(
        id=None,
        raw_file='/tmp/v1.wav'
        # raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00254I0026.wav',
    )

    src_mel = src_uttrn.melspectrogram()
    
    src_embed = speaker_encoder(torch.unsqueeze(torch.from_numpy(src_mel), 0).transpose(1, 2).cuda())
    # src_mel = pad_zeros(src_mel)
    src_mels = torch.unsqueeze(torch.from_numpy(src_mel), 0).transpose(1, 2).cuda()

    # 804 female sharp
    # 1 female soft
    # tgt_uttrns = speakers[1].random_utterances(10)
    # print(f'tgt raw file {tgt_uttrns[0].raw_file}')
    # tgt_uttrns = [dataset.Utterance(id=None, raw_file=f'/tmp/a{i}.wav') for i in range(1, 5)]

    tgt_uttrns = [
        dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00254I0026.wav'),
        dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00254I0027.wav'),
        dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00254I0028.wav'),
        dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00254I0029.wav'),
        dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00254I0030.wav'),

        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0030.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0031.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0032.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0033.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0034.wav'),

        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0025.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0026.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0027.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0028.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS/20170001P00047I0029.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00211I0107.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00211I0060.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00211I0061.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00211I0062.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00211I0063.wav'),
        # dataset.Utterance(id=None, raw_file='/mnt/ssd500/dataset/speech/ST_CMDS_holdout/20170001P00211I0064.wav'),
    ]
    tgt_mels = [tgt_uttrn.melspectrogram() for tgt_uttrn in tgt_uttrns]

    tgt_embeds = []
    for m in tgt_mels:
        tgt_embeds.append(speaker_encoder(torch.from_numpy(m).unsqueeze(0).transpose(1, 2).cuda()))
    tgt_embed = torch.cat(tgt_embeds, dim=0).unsqueeze(0)
    # tgt_embed = speaker_encoder(torch.from_numpy(np.array(tgt_mels)).transpose(1, 2).cuda()).mean(dim=0, keepdim=True) # S2

    print(f'src_mels {src_mels.shape}')
    print(f'src_embed {src_embed.shape}')
    print(f'tgt_embed {tgt_embed.shape}')

    init_out, out_mels, content_out, _ = generator(src_mels, src_embed, tgt_embed)
    init_out2, out_mels2, content_out2, _ = generator(src_mels, src_embed, src_embed.unsqueeze(1))

    # loss, recon_loss, recon0_loss, content_recon_loss = generator.loss(src_mels,
    #                 src_embed,
    #                 init_out,
    #                 out_mels,
    #                 content_out)

    # logging.info(f'inference loss {loss:.3f} recon_loss {recon_loss:.3f} recon0_loss {recon0_loss:.3f} content_recon_loss {content_recon_loss:.3f}')

    netG = model_vocoder.Generator(hp.num_mels, hp.vocoder_ngf, hp.vocoder_n_residual_layers).cuda()
    ckpts = sorted(list(Path(hp.vocoder_save_dir).glob('*.pt')))
    if len(ckpts) > 0:
        latest_ckpt_path = ckpts[-1]
        logging.info(f'loading vocoder ckpt {latest_ckpt_path}')
        ckpt = torch.load(latest_ckpt_path)
        netG.load_state_dict(ckpt['netG_state_dict'])
    S = out_mels.squeeze(1).transpose(1, 2)
    y_recon = netG(src_mels.transpose(1, 2))
    y_pred = netG(S)
    y_recon2 = netG(out_mels2.squeeze(1).transpose(1, 2))
    print(f'shapes out_mels {out_mels.shape}, S {S.shape}, y_pred {y_pred.shape}')

    results = [
        src_mels.detach().cpu().numpy(),
        tgt_mels,
        out_mels.detach().cpu().numpy(),
        y_pred.detach().cpu().numpy(),
        y_recon.detach().cpu().numpy(),
        src_uttrn.raw(sr=hp.sample_rate),
        tgt_uttrns[0].raw(sr=hp.sample_rate),
        out_mels2.detach().cpu().numpy(),
        y_recon2.detach().cpu().numpy(),
    ]

    with open('generator_results.pkl', 'wb') as f:
        pickle.dump(results, f)

def trace():
    generator = Generator(8, 256, 512, 4)
    if args.model:
        ckpt = torch.load(args.model)
        if ckpt:
            logging.info(f'loading generator ckpt {args.model}')
            generator.load_state_dict(ckpt['model_state_dict'])
        
    else:
        ckpts = sorted(list(Path(hp.generator_save_dir).glob('*.pt')))
        if len(ckpts) > 0:
            latest_ckpt_path = ckpts[-1]
            ckpt = torch.load(latest_ckpt_path)
            if ckpt:
                logging.info(f'loading generator ckpt {latest_ckpt_path}')
                generator.load_state_dict(ckpt['model_state_dict'])

    device = torch.device("cpu")
    generator.to(device=device)
    generator.eval()

    x1 = torch.ones(1, 298, 80)
    x2 = torch.ones(1, 256)
    x3 = torch.ones(1, 10, 256)

    # out = generator(x1, x2, x3)

    enc_x_1 = torch.ones(1, 320, 80)
    enc_x_2 = torch.ones(1, 256)
    # dec_x = torch.ones(1, 256, 32*2+256)
    post_x = torch.ones(1, 80, 298)
    # out = generator(x1, x2, x3)
    traced_postnet = torch.jit.trace(generator.postnet, (post_x))
    generator.postnet = traced_postnet
    sm = torch.jit.script(generator, (x1, x2, x3))
    print(sm.code)
    out = sm(x1, x2, x3)
    print(out.shape)
    print(out)
    sm.save('autovc_script_model.pt')

if __name__ == '__main__':
    if args.train:
        train()
    elif args.inference:
        inference()
    elif args.trace:
        trace()
    else:
        print('nothing to do')
