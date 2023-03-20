# Paper review: Demosaicing to Detect Demosaicing and Image Forgeries

Author of the code and the review: Jérémie Dentan
Authors of the original paper: Quentin Bammey, Rafael Grompone von Gioi, Jean-Michel Morel

## Run the code

The code of this repository is expected to run in **Python 3.9** with the dependencies of `requirements.txt` installed and your PYTHONPATH set to the root of the repository. To do so, execute the following line from the root (i.e. from the folder containing this README file).

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

Moreover, if you want to execute locally the notebooks in the `\doc` folder, you should have `ipykernel` installed, which is not declared in `requirements.txt` for technical reasons linked to the possibility to execute the notebooks on Google Colab. Thus, you should run the following before running the notebooks:

```bash
pip install notebook==6.5.3
```
