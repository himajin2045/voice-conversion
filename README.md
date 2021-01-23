### Voice Conversion

This is a collection of speech models including a speaker encoder model, a voice conversion model and a vocoder model together as a complete Voice Conversion pipeline.

The sepaker encoder model is an implementation make use of the GE2E loss and the code is from https://github.com/CorentinJ/Real-Time-Voice-Cloning

The voice conversion model is AutoVC and the code is from https://github.com/auspicious3000/autovc

The vocoder model is https://github.com/descriptinc/melgan-neurips

**You must make appropriate changes (e.g. change the path of datasets, the model parameters in hparams.py or the train_xxx.py files) in order to run the code. I'm not going to explain the code since it's almost been a year since the last time I run the code, I don't remember the details :)**
