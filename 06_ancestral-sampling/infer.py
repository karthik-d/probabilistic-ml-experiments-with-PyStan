import os
import pystan
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plot

MODEL_PATH = os.getcwd()
MODEL_SRCNAME = 'model.stan'
INFERENCE_FILENAME = 'vectorized_inference.pkl'
MODEL_FILENAME = 'model.pkl'
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'Data')  # Parent directory
RESPONSE_NAME = 'responses.csv'
SKILLS_NEEDED_NAME = 'skills_needed.csv'

def plotColorMap(data):
    rows = len(data)
    cols = len(data[0])
    fig, ax = plot.subplots(1, 1, tight_layout=True)
    for x in range(rows + 1):
        ax.axhline(x, lw=1, color='black', zorder=5)
    for x in range(cols+1):
        ax.axvline(x, lw=1, color='black', zorder=5)
    ax.imshow(data, interpolation='none', cmap=plot.get_cmap('gray'), extent=[0, cols, 0, rows], zorder=0)
    ax.axis('off')
    plot.show()

def readSkillsNeeded():
    convert = {'True':1, 'False':0}
    skills_needed= list()
    with open(os.path.join(DATA_PATH, SKILLS_NEEDED_NAME)) as f:
        f_reader = csv.reader(f)
        for data_row in f_reader:
            curr = list()
            for value in data_row:
                curr.append(convert[value])
            skills_needed.append(curr)
    return skills_needed

'''
data_dict = ('isCorrect1', 'isCorrect2', 'isCorrect3')
allowed_values = (0, 1)
observed_data = [dict(zip(data_dict,
                        [(a,b,c) for a in allowed_values for b in allowed_values for c in allowed_values][x])) for x in range(len(allowed_values)**len(data_dict))]
'''

'''
int num_questions;
int num_skills;
int<lower=0,upper=1> isCorrect[num_questions] ;
matrix<lower=0,upper=1>[num_questions, num_skills] skillsNeeded;

num_questions = 3
num_skills = 2
num_samples = 10
'''

''' ______________________________________ ANCESTRAL SAMPLING ________________________ '''

skills_needed = readSkillsNeeded()
observed_data = [{'num_questions':len(skills_needed), 'num_skills':len(skills_needed[0]), 'skillsNeeded':skills_needed}]

#Compilation of model
try:
    model_file_path = os.path.join(MODEL_PATH, MODEL_FILENAME)
    model_file = open(model_file_path, 'rb')
except FileNotFoundError:
    model_source_path = os.path.join(MODEL_PATH, MODEL_SRCNAME)
    with open(model_source_path) as model_source:
        model_code = model_source.read()
    model = pystan.StanModel(model_code=model_code, verbose=False)

    # Save the compiled model to avoid recompilation
    # Explicitly delete the *.pkl file if model has been modified
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
else:
    model = pickle.load(model_file)
    model_file.close()

num_samples = 22
samples = {'skills':list(), 'isCorrect':list()}
for data in observed_data:
    result = model.sampling(data=data, algorithm='Fixed_param', chains=1, iter=num_samples)
    samples_raw = result.extract(pars=('skills', 'isCorrect'), permuted=True)
    samples['skills'] = samples_raw['skills']
    samples['isCorrect'] = samples_raw['isCorrect']
    #print(result.extract(pars=('isCorrect')))
    #inference['skill'].append(summary['summary'][0])


''' ______________________________________ INFERENCE FROM SAMPLED VALUES _______________________________ '''

ground_truth = samples['skills']
print(ground_truth)
is_correct = list()
for case in samples['isCorrect']:
    is_correct.append(list(map(lambda x:int(x), case)))

'''
data_dict = ('isCorrect1', 'isCorrect2', 'isCorrect3')
allowed_values = (0, 1)
observed_data = [dict(zip(data_dict,
                        [(a,b,c) for a in allowed_values for b in allowed_values for c in allowed_values][x])) for x in range(len(allowed_values)**len(data_dict))]
'''

observed_data = list()
num_questions = len(skills_needed)
num_skills = len(skills_needed[0])
for case in is_correct:
    observed_data.append({'num_questions':num_questions, 'num_skills':num_skills, 'isCorrect':case, 'skillsNeededArr':skills_needed})

#Compilation of model
try:
    model_file_path = os.path.join(MODEL_PATH, INFERENCE_FILENAME)
    model_file = open(model_file_path, 'rb')
except FileNotFoundError:
    print("Pickled Inference Model could not be found!")
    exit()
else:
    model = pickle.load(model_file)
    model_file.close()

inference = list()   # Matrix of skill_probabilities for each observed_data

for data in observed_data:
    result = model.sampling(data=data, pars=('P_skills'))
    curr = list()
    for k in range(num_skills):
        curr.append(result.summary(pars=('P_skills'))['summary'][k][0])
    inference.append(curr)            # Mean Probabilties of All Skills
    print(inference)

for k in inference:
    print(k)
plotColorMap(inference)
plotColorMap(ground_truth)
