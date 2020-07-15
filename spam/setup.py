# nsml: registry.navercorp.com/abuse-filtering-public/ai-rush-baseline:latest
from distutils.core import setup

setup(
    name="Spam classification - AI Rush baseline",
    version="1",
    install_requires=['cachetools==4.1.0',
                      'fire==0.3.1',
                      'h5py==2.7.1',
                      'matplotlib==3.2.2',
                      'scikit-learn==0.23.1',
                      'keras==2.3.1',
                      'keras_applications==1.0.8',
                      'keras_metrics==1.1.0',
                      'numpy==1.19.0',
                      'pandas==1.0.5',
                      'pillow==7.1.2',
                      'requests==2.24.0',
                      'tqdm==4.46.1',
                      'wget==3.2']
)
