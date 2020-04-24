# TTS-ROBIN

This project is a RunTime Optimization for TTS-Cube (https://github.com/tiberiu44/TTS-Cube). Only models trained with TTS-Cube can be used here.

The vocoder used here is ClariNet (https://github.com/ksw0306/ClariNet).

# Prerequisites

1. Python 3
2. DyNet
3. PyTorch
4. NumPy
5. SciPy
6. Flask

ClariNet is already provided in the project.

# How to run

To run a model, add the .network files on "models" folder.

To test off-line, add sentences in "texts.txt" file. Every line will be synthethised on a file in "tests" folder. The command is ```python Test.py```

To test on-line, run the command ```python WebService.py --port PORT_ID```. A flask server will start on port PORT_ID (default is 8080). The file "TestWebService.py" presents an example on how to use the server. Note that if you use the server over network, you need to add the IP of the host (the computer that runs the server) instead of 127.0.0.1.
