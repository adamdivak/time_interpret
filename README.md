# On the reproducibility of: "Learning Perturbation to Explain Time Series Predictions"
### Authors: Wouter Bant, Ádám Divák, Jasper Eppink, and Floris Six Dijkstra
Paper and reviews: [https://openreview.net/forum?id=nPZgtpfgIx](https://openreview.net/forum?id=nPZgtpfgIx)

<p align="center">
  <img src="assets/explained_saliency.svg">
</p>

 We used the provided code of [Joseph Enguehard](https://github.com/josephenguehard/time_interpret) to run all experiments. 
 
 Some additions/changes we made:
- Adding a [notebook that shows the main results we obtained](results/main.ipynb) and [a notebook that visualizes the learned masks and perturbations](results/saliency_perturbation_debug_plots.ipynb).
- Fixing an issue in the implementation of the deletion game ([commit](https://github.com/yosuah/time_interpret_private/commit/a9b77f1fdc82e4157a059da5d3959f34ffb48818)).
- Adjusting the loss function for the deletion game ([commit](https://github.com/yosuah/time_interpret_private/commit/9693264fccf0a99ea1732fcd070b9dbc8e166955)).
- Providing code for experiments on a weather dataset ([commit](https://github.com/yosuah/time_interpret_private/commit/e806bb78dd2c5337e58de80b4d6a58caf40cc3ed)).
- Fix HMM data caching issue, causing incorrect results in the paper ([commit](https://github.com/yosuah/time_interpret_private/commit/cfe87ad91bc186dc8c12ccfc0395d2fbd424e9dc)).
- Providing additional options to run the revised deletion game easily.
- Adding code that saves files used for certain tables/figures.

## Getting started

### Installing the dependencies
Clone the repository:
```
git pull https://github.com/yosuah/time_interpret.git
```
Go inside the root directory:
```
cd time_interpret
```
Create the conda environment
```
conda env create -f environment.yml
```

### Running one fold
Activate the conda environment:
```
conda activate tint
```

You may need to specify a PYTHONPATH, on Linux:
```
export PYTHONPATH=/home/<username>/tint/time_interpret_private:$PYTHONPATH
```

To run the hmm experiment:
```
python experiments/hmm/main.py
```

To run the MIMIC-III experiment (Note that you need credentialized access to this datatset):
```
python experiments/mimic3/mortality/main.py
```

To run the weather data experiment:
```
python experiments/weather/main.py
```

If you don't want to run all explanation methods you can add a flag such as:
```
--explainers extremal_mask_preservation extremal_mask_deletion dyna_mask deep_lift
```

## Acknowledgment
- [Joseph Enguehard](https://github.com/josephenguehard/time_interpret) for almost all the code.
