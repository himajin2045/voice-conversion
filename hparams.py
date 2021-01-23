from pathlib import Path

###  general

seed = 1234
log_path = '/mnt/ssd500/speech/autovc/log.txt'


### audio

min_level_db = -100
sample_rate = 16000
n_fft = 1024
num_mels = 80
fmin = 90
fmax = 7600
hop_length = 256
win_length = 1024
ref_level_db = 16


### dataset

# mel_feature_root = Path('/mnt/ssd500/speech/autovc/data')
# mel_feature_root = Path('/mnt/ssd500/speech/autovc/data_80_mels')
mel_feature_root = Path('/mnt/ssd500/speech/autovc/data_80_mels')

stcmds_data_root = Path('/mnt/ssd500/dataset/speech/ST-CMDS-20170001_1-OS')
aishell_data_root = Path('/mnt/ssd500/dataset/speech/aishell/data_aishell/wav')
aidatatang_data_root = Path('/mnt/ssd500/dataset/speech/aidatatang_200zh/corpus')
primewords_data_root = Path('/mnt/ssd500/dataset/speech/primewords/extract/primewords_md_2018_set1')
toy_data_root = Path('/mnt/ssd500/dataset/speech/toy')


### toy

# speakers_per_batch=2
# utterances_per_speaker=5
# seq_len=130
# train_steps = 1e12
# train_print_interval = 10 # in steps
# total_evaluate_steps = 20
# evaluate_interval =  50 # in steps
# save_interval = 100 # in steps
# save_dir = '/mnt/ssd500/speech/autovc/ckpts'
# max_ckpts = 3


### speaker encoder

speakers_per_batch=64
utterances_per_speaker=10
seq_len=128
train_steps = 1e12
train_print_interval = 100 # in steps
total_evaluate_steps = 50
evaluate_interval = 500 # in steps
save_interval = 500 # in steps
save_dir = '/mnt/ssd500/speech/autovc/speaker_ckpts'
max_ckpts = 30
speaker_lr = 1e-4


### generator

generator_train_print_interval = 50 # in steps
generator_total_evaluate_steps = 500
generator_evaluate_interval = 1000 # in steps
generator_save_interval = 1000 # in steps
generator_speakers_per_batch = 16 # 4
generator_utterances_per_speaker = 1
generator_save_dir = '/mnt/ssd500/speech/autovc/generator_ckpts'
generator_train_steps = 1e12
generator_max_ckpts = 5
generator_lr = 1e-4
generator_seq_len = 128
def generator_get_lr(step):
    if step < 300000:
        return 1e-4
    if step < 400000:
        return 5e-5
    if step < 500000:
        return 1e-5
    else:
        return 1e-6
    # if step < 300000:
    #     return 1e-4
    # elif step < 350000:
    #     return 5e-5
    # elif step < 400000:
    #     return 1e-5
    # else:
    #     return 1e-6
generator_sample_interval = 200
generator_bak_interval = 100000
# generator_bak_dir = '/mnt/ssd500/speech/autovc/g10_ckpts_bak'


### vocoder

vocoder_seq_len = 8192
vocoder_train_steps = 1e12
vocoder_G_lr = 1e-4
vocoder_D_lr = 1e-4
vocoder_n_layers_D = 4
vocoder_num_D = 3
vocoder_ngf = 32
vocoder_n_residual_layers = 3
vocoder_ndf = 16
vocoder_downsamp_factor = 4
vocoder_batch_size = 2
vocoder_save_dir = '/mnt/ssd500/speech/autovc/vocoder_ckpts'
vocoder_train_print_interval = 100
vocoder_save_interval = 1000
vocoder_max_ckpts = 20


### adagan
adagan_batch_size = 4
adagan_seq_len = 128
adagan_train_steps = 1e12
adagan_G_lr = 1e-4
adagan_D_lr = 1e-4
adagan_save_dir = '/mnt/ssd500/speech/autovc/adagan_ckpts'
adagan_train_print_interval = 100
adagan_save_interval = 1000
adagan_max_ckpts = 20
adagan_sample_interval = 500


### stargan2
stargan2_seq_len = 128
stargan2_batch_size = 1
stargan2_G_lr = 2e-4
stargan2_D_lr = 2e-4
stargan2_save_dir = '/mnt/ssd500/speech/autovc/stargan2_ckpts'
stargan2_train_steps = 1e12
stargan2_train_print_interval = 10
stargan2_save_interval = 100
stargan2_max_ckpts = 5
stargan2_sample_interval = 100


### melganvc
melganvc_seq_len = 128
melganvc_batch_size = 2
melganvc_G_lr = 1e-4
melganvc_D_lr = 1e-4
melganvc_S_lr = 1e-4
melganvc_save_dir = '/mnt/ssd500/speech/autovc/melganvc_ckpts'
melganvc_train_steps = 1e12
melganvc_train_print_interval = 50
melganvc_save_interval = 1000
melganvc_max_ckpts = 5
melganvc_sample_interval = 100


### adagan2
adagan2_batch_size = 8
adagan2_seq_len = 128
adagan2_train_steps = 1e12
adagan2_G_lr = 1e-4
adagan2_D_lr = 2e-4
adagan2_save_dir = '/mnt/ssd500/speech/autovc/adagan2_ckpts'
adagan2_train_print_interval = 100
adagan2_save_interval = 1000
adagan2_max_ckpts = 20
adagan2_sample_interval = 500

### infogan 
infogan_batch_size = 2
infogan_seq_len = 128
infogan_train_steps = 1e12
infogan_G_lr = 2e-4
infogan_D_lr = 4e-4
infogan_save_dir = '/mnt/ssd500/speech/autovc/infogan_ckpts'
infogan_train_print_interval = 100
infogan_save_interval = 1000
infogan_max_ckpts = 20
infogan_sample_interval = 500


### vocoder tf
vocoder_tf_summary_path = '/mnt/ssd500/speech/autovc/vocoder_tf_summaries/v3'
vocoder_tf_ckpt_path = '/mnt/ssd500/speech/autovc/vocoder_tf_ckpts/v3'
