MediaEval AcousticBrainz Genre Task Feature Baseline reimplementation
===

Setup
---
Download the development (training) and validation datasets into the `data` directory.
The datasets are located at https://zenodo.org/record/2553414 and https://zenodo.org/record/2554044
(see https://multimediaeval.github.io/2018-AcousticBrainz-Genre-Task/data for further information).

Install `parallel`.

`sudo apt install parallel`

Run `process.sh` to uncompress the data.

`./process.sh`

Install the required python packages (in a virtualenv).

`pip install -r requirements.txt`

Execute `preprocessing.py` to run the preprocessing steps.

`./preprocessing.py`


References
---
https://archives.ismir.net/ismir2019/paper/000042.pdf
