import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    # 1xtimexfreq, 1x256
    def forward(self, x, c_org):
        # print(f'encoder x.shape {x.shape}, c_org.shape {c_org.shape}')
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1) # nx(80+256)xtime
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        # print(f'encoder conv outputs.shape {x.shape}')

        # x: nx512xseq_len 

        x = x.transpose(1, 2) # nxseq_lenx512
        
        # self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        # outputs: nxseq_lenx(dim_neckx2) -> nx128x64
        # print(f'encoder lstm outputs.shape {outputs.shape}')
        out_forward = outputs[:, :, :self.dim_neck]
        # out_forward: nx128x32
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            code = torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1)
            # code: nx64 <- cat(nx32, nx32)
            # print(f'code shape {code.shape}')
            codes.append(code)

        # print(f'codes len {len(codes)}')
        # len(codes): 4 == 128 // 32 == seq_len // freq
        return torch.stack(codes, dim=0) # 4xnx64
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    # 32, 256, 512
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()

        # x: nxseq_lenx(dim_neck*2+dim_emb)

        x, _ = self.lstm1(x)

        # x: nxseq_lenxdim_pre

        x = x.transpose(1, 2)

        # x: nxdim_prexseq_len
        
        for conv in self.convolutions:
            x = F.relu(conv(x))

        # x: nxdim_prexseq_len

        x = x.transpose(1, 2)

        # x: nxseq_lenxdim_pre
        
        outputs, _ = self.lstm2(x)

        # outputs: nxseq_lenx1024
        
        decoder_output = self.linear_projection(outputs)

        # print(f'decoder_output: {decoder_output.shape}')
        # decoder_output: nxnxseq_lenx80

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """Generator network."""
    # 32, 256, 512, 32
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        self.freq = freq
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    # def l1_loss(self, x, y):
    #     return torch.mean(torch.sum(torch.abs(x-y), dim=1))

    # def l2_loss(self, x, y):
    #     return torch.mean(torch.norm((x-y), dim=(1, 2)))

    # nxtimexfreq, nx256, nxkx256
    def forward(self, x, c_org, c_trg):
        c_trg = torch.mean(c_trg, dim=1) # nx256

        # x = 20 * torch.log10(torch.clamp(x, 1e-5))
        # x = torch.clamp((x + 100) / 100, 0, 1)

        residual = x.size(1) % self.freq
        if residual > 0:
            pad_len = self.freq - residual
            x = F.pad(x, (0, 0, 0, pad_len), mode='constant')
                
        codes = self.encoder(x, c_org)

        # print(f'codes shape {codes.shape}') # (16, n, 64)

        tmp = []
        for i in range(codes.size(0)):
            # print(f'code {code.shape}')
            # code: nx64
            c_tmp = codes[i].unsqueeze(1).expand(-1,int(x.size(1)/codes.size(0)),-1)
            # c_tmp: nxfreqx64
            tmp.append(c_tmp)
        code_exp = torch.cat(tmp, dim=1)
        # code_exp: nx128x64
        # print(f'code_exp {code_exp.shape}')

        # print(f'code_exp {code_exp.shape} c_trg {c_trg.shape}')
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        # print(f'encoder_outputs {encoder_outputs.shape}')
        # encoder_outputs: nxseq_lenx(neck_dimx2+emb_dim)

        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)

        flat_codes = codes.permute(1, 0, 2).contiguous()
        flat_codes = flat_codes.view(flat_codes.size(0), flat_codes.size(1) * flat_codes.size(2))
        
        # return mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet, flat_codes, code_exp

    def loss(self, src_mels, src_embeds, init_out, final_out, content_out):
        init_out = init_out.squeeze(1)
        final_out = final_out.squeeze(1)
        recon0_loss = F.l1_loss(src_mels, init_out)
        recon_loss = F.l1_loss(src_mels, final_out)

        # with open('content.pkl', 'wb') as f:
        #     pickle.dump([src_mels.detach().cpu().numpy(), init_out.detach().cpu().numpy(), final_out.detach().cpu().numpy()], f)

        codes = self.encoder(final_out, src_embeds)
        flat_codes = codes.permute(1, 0, 2).contiguous()
        content_recon_out = flat_codes.view(-1, flat_codes.size(1) * flat_codes.size(2))
        content_recon_loss = F.l1_loss(content_out, content_recon_out)
        # import pickle
        # with open('content.pkl', 'wb') as f:
        #     pickle.dump([content_out.detach().cpu().numpy(), content_recon_out.detach().cpu().numpy()], f)
        total_loss = recon_loss + 1. * recon0_loss + 1. * content_recon_loss
        return total_loss, recon_loss, recon0_loss, content_recon_loss

    
if __name__ == '__main__':
    net = Generator(32, 256, 512, 32)
    spectro = torch.randn(8, 128, 80)
    orig_speaker_emb = torch.randn(8, 256)
    tgt_speaker_emb = torch.randn(8, 256)
    _, out, _ = net(spectro, orig_speaker_emb, tgt_speaker_emb)
    print(f'out.shape {out.shape}')
