# testTask

if you are looking for metrics results:
far and frr are 9/9 and 9/15 

My working framework was pytorch. I`ve used this guide a lot: https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
1. Please take a note that not all samples from train and test are considered in testing, training or validation 
due to duration<=5, duration >=3 constraint
2. Also, there was no acoustic model, the model just used raw waveform and conv layers as a way of user identification.
3. As a rejection i`ve implemented just a simple check : (tensor.argmax()) < 0 or not. 
