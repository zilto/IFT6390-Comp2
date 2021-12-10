import numpy as np
import pandas as pd

            
def study_summary(study):
    # function to display optuna's best parameters
    # not all parameters are displayed, just those from the grid
    print("Study: ", study.study_name)
    print("\nNumber of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Validation score: {}".format(trial.value))
    print("  Params: ")
    for key, val in trial.params.items():
        print("    {}: {}".format(key, val))