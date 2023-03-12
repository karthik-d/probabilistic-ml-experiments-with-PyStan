import pystan
import pickle
import os

MODEL_PATH = os.getcwd()
MODEL_SRCNAME = 'model.stan'
MODEL_FILENAME = 'model.pkl'

# GENERATE ALL COMBINATION OF SCORES

data_dict = ('isCorrect1', 'isCorrect2', 'isCorrect3', 'isCorrect4')
allowed_values = (0, 1)
observed_data = [dict(zip(data_dict,
                        [(a,b,c,d) for a in allowed_values for b in allowed_values for c in allowed_values for d in allowed_values][x])) for x in range(len(allowed_values)**len(data_dict))]

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

inference = {'skill1':list(), 'skill2':list()}
for data in observed_data:
    result = model.sampling(data=data)
    #print(result)
    summary = result.summary(pars=('P_skill1', 'P_skill2', 'skill1'))
    inference['skill1'].append(summary['summary'][0][0])
    inference['skill2'].append(summary['summary'][1][0])
    print(summary['summary'][2][0])
    print(result)

    """plot.hist(result['skill1'], bins=[x*0.01 for x in range(0,101)], rwidth=0.9)
    plot.title('Probability Distribution')
    plot.show()

    plot.hist(result['skill2'], bins=[x*0.01 for x in range(0,101)], rwidth=0.9)
    plot.title('Probability Distribution')
    plot.show()

    plot.hist(result['hasSkills'], bins=[x*0.01 for x in range(0,101)], rwidth=0.9)
    plot.title('Probability Distribution')
    plot.show()"""

print()
for k in observed_data:
    print(k)
print("\nSKILL 1".ljust(24), "SKILL 2".ljust(24))
for i in range(len(inference['skill1'])):
    print(str(inference['skill1'][i]).ljust(24), str(inference['skill2'][i]).ljust(24))
