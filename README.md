### Voice Conversion

This is a collection of speech models including a speaker encoder model, a voice conversion model and a vocoder model together as a complete Voice Conversion pipeline.

The sepaker encoder model is an implementation make use of the GE2E loss and the code is from https://github.com/CorentinJ/Real-Time-Voice-Cloning

The voice conversion model is AutoVC and the code is from https://github.com/auspicious3000/autovc

The vocoder model is https://github.com/descriptinc/melgan-neurips

**You must make appropriate changes (e.g. change the path of datasets, the model parameters in hparams.py or the train_xxx.py files) in order to run the code. I'm not going to explain the code since it's almost been a year since the last time I run the code, I don't remember the details :)**

The workflow is quite simple though:

1. You collect some speech datasets (see dataset.py)
2. Then run the preprocess.py script to convert raw audios to mel features so we don't need to do the conversion on the fly while training
3. Train the speaker encoder model
4. Train the vocoder model
5. Train the voice conversion model, this depends on a trained speaker encoder model
6. Run the inference function in the train_vc.py script to do the conversion, this depends on all 3 models

The speaker encoder model and the vocoder model should be trained on a large dataset combined from many corpus (see dataset.py for all corpus I used).

The voice conversion model could be trained on a small number of speakers from one of the corpus, 120 speakers and 120 utterances per speaker is good enough to get sound performance.
