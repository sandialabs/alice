# Instructions for setting up virtual environment
- Ensure that Python 3.11, Cuda 11.8, and GCC 8.5 are all available.
- python -m venv venv_alice
- . venv_alice/bin/activate
- pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
- pip install --upgrade setuptools
- pip install wheel
- pip install . --verbose --no-build-isolation

# To test the installation of alice
- cd experiments/resnet18
- . irn18.sh

# To run some tests, the following packages must also be installed.
- pip install imageio
- pip install pytorch_optimizer

