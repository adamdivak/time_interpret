# On the reproducibility of: "Learning Perturbation to Explain Time Series Predictions"
### Authors: Wouter Bant, Ádám Divák, Jasper Eppink, and Floris Six Dijkstra
Paper and reviews: [https://openreview.net/forum?id=nPZgtpfgIx](https://openreview.net/forum?id=nPZgtpfgIx)

*Deep Learning models have taken the front stage in the AI community, yet explainability challenges hinder their widespread adoption. Time series models, in particular, lack attention in this regard. This study tries to reproduce and extend the work of Enguehard (2023b), focusing on time series explainability by incorporating learnable masks and perturbations. Enguehard (2023b) employed two methods to learn these masks and perturbations, the preservation game (yielding SOTA results) and the deletion game (with poor performance). We extend the work by revising the deletion game’s loss function, testing the robustness of the proposed method on a novel weather dataset, and visualizing the learned masks and perturbations. Despite notable discrepancies in results across many experiments, our findings demonstrate that the proposed method consistently outperforms all baselines and exhibits robust performance across datasets. However, visualizations for the preservation game reveal that the learned perturbations primarily resemble a constant zero signal, questioning the importance of learning perturbations. Nevertheless, our revised deletion game shows promise, recovering meaningful perturbations and, in certain instances, surpassing the performance of the preservation game.*

<p align="center">
  <img src="assets/explained_saliency.svg">
</p>

 We used the provided code of [Joseph Enguehard](https://github.com/josephenguehard/time_interpret) to run all experiments. 
 
 Some additions/changes we made:
- Adding a [notebook that shows the main results we obtained](results/main.ipynb) and [a notebook that visualizes the learned masks and perturbations](results/saliency_perturbation_debug_plots.ipynb).
- Fixing an issue in the implementation of the deletion game ([commit](https://github.com/yosuah/time_interpret/commit/a9b77f1fdc82e4157a059da5d3959f34ffb48818)).
- Adjusting the loss function for the deletion game ([commit](https://github.com/yosuah/time_interpret/commit/9693264fccf0a99ea1732fcd070b9dbc8e166955)).
- Providing code for experiments on a weather dataset ([commit](https://github.com/yosuah/time_interpret/commit/e806bb78dd2c5337e58de80b4d6a58caf40cc3ed)).
- Fix HMM data caching issue, causing incorrect results in the paper ([commit](https://github.com/yosuah/time_interpret/commit/cfe87ad91bc186dc8c12ccfc0395d2fbd424e9dc)).
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
export PYTHONPATH=/home/<username>/tint/time_interpret:$PYTHONPATH
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
