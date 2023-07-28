import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols, romans
from text import text_to_phoneme, phoneme_to_sequence

from scipy.io.wavfile import write

import soundfile as sf

from transformers import T5ForConditionalGeneration, AutoTokenizer
from phonemizer import phonemize

from text.mapping_table import similar_ipa



# model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
# tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

# def get_IPA(text, lang):
#     words = text.split()
#     words = ["<{}>".format(lang) + ": " +i for i in words]
#     out = tokenizer(words,padding=True,add_special_tokens=False,return_tensors='pt')
#     preds = model.generate(**out,num_beams=1,max_length=50)
#     phones = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)
#     new_IPA = []
#     for phone in phones:
#         _phone = [phone]
#         phone = " ".join(_phone)
#         new_ipa = similar_ipa(phone)
#         new_ipa = [i for i in new_ipa.split(" ")]
#         new_ipa = " ".join(new_ipa)
#         new_IPA.append(new_ipa)
#     new_IPA = "  ".join(new_IPA)
#     return new_IPA



# def get_IPA(text, lang):
#    IPA = phonemize(text, language=lang, backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
#    _IPA = [i for i in IPA]
#    _IPA = " ".join(_IPA)
#    new_IPA = similar_ipa(_IPA, lang)
#    new_IPA = new_IPA.replace("   ", "_")
#    new_IPA = new_IPA.replace(" ", "")
#    new_IPA = new_IPA.replace("_", " ")
#    return new_IPA


def get_text(text, hps, type_of_phoneme, lang):
    phoneme = text_to_phoneme(text,type_of_phoneme, lang)
    print(phoneme)
    text_norm = phoneme_to_sequence(phoneme, type_of_phoneme) 
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# load pre-trained model
def load_pretrained(type_of_phoneme):
    if type_of_phoneme == "IPA":
        hps = utils.get_hparams_from_file("./configs/ljs_base.json")

        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cuda()
        _ = net_g.eval()

        _ = utils.load_checkpoint("./checkpoints/pretrained_ljs.pth", net_g, None)
    
    elif type_of_phoneme == "uroman":
        ckpt_dir = "/home/solee0022/tts-asr/eng"
        config_file = f"{ckpt_dir}/config.json"
        assert os.path.isfile(config_file), f"{config_file} doesn't exist"
        hps = utils.get_hparams_from_file(config_file)
        #text_mapper = TextMapper(vocab_file)
        net_g = SynthesizerTrn(
            len(romans),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        net_g.to(device)
        _ = net_g.eval()

        g_pth = f"{ckpt_dir}/G_100000.pth"
        _ = utils.load_checkpoint(g_pth, net_g, None)

    return net_g



def synthesis_inference(path, text, type_of_phoneme, lang):
    stn_tst = get_text(path, text, type_of_phoneme, lang) 

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        # phoneme type에 따라 불러오는 model checkpoint가 다름
        net_g = load_pretrained(type_of_phoneme)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    sf.write('/home/solee0022/tts-asr/synthetic_speech/vits/ko3/' + path, audio, hps.data.sampling_rate)
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))


f = open("/home/solee0022/data/cv/ko/train.txt")
lines = f.readlines()

for line in lines[:10]:
    path, text = line.split("|")
    synthesis_inference(path, text, "uroman", "ko")



