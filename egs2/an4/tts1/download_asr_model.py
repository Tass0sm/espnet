from espnet2.bin.asr_inference import Speech2Text

speech2text = Speech2Text.from_pretrained(
    # "kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best",
    "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1,
)

print(speech2text)
