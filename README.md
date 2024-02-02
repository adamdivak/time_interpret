# On the reproducibility of: "Learning Perturbation to Explain Time Series Predictions"
### Authors: Wouter Bant, Ádám Divák, Jasper Eppink, and Floris Six Dijkstra
 We used the provided code to run all experiments. See the commits to see the changes we made. 
 
 These changes can be summarized as:
- Fixing an issue in the implementation of the deletion game.
- Adjusting the loss function for the deletion game.
- Providing code for experiments on an weather dataset.
- Providing a folder with the results and a notebook that displays these results.
- Providing additional options to easily run the revised deletion game.
- Adding code that saves files that are used for certain tables/figures.

## Acknowledgment
- [Joseph Enguehard](https://github.com/josephenguehard/time_interpret) for almost all the code.

## Getting started

### Installing the dependencies
Clone the repository:
```
git pull https://github.com/yosuah/time_interpret_private.git
```
Go inside the root directory:
```
cd time_interpret_private
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

