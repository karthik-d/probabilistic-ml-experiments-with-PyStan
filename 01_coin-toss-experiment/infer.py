import os
import pystan
import pickle
from matplotlib import pyplot as plot

MODEL_PATH = os.getcwd()
MODEL_SRCNAME = 'model.stan'
MODEL_FILENAME = 'model.pkl'

observed_data = { 'N_heads':3,
                  'N_tails':0}

#Compilation of model
try:
    model_file_path = os.path.join(MODEL_PATH, MODEL_FILENAME)
    model_file = open(model_file_path, 'rb')
except FileNotFoundError:
    model_source_path = os.path.join(MODEL_PATH, MODEL_SRCNAME)
    with open(model_source_path) as model_source:
        model_code = model_source.read()
    model = pystan.StanModel(model_code=model_code)

    # Save the compiled model to avoid recompilation
    # Explicitly delete the *.pkl file if model has been modified
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
else:
    model = pickle.load(model_file)
    model_file.close()

result = model.sampling(data=observed_data)
print(result.stansummary())

plot.hist(result['theta'], bins=[x*0.01 for x in range(0,101)], rwidth=0.9)
plot.title('Probability Distribution')
plot.show()
