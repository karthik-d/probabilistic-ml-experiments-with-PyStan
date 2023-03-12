import os
import pystan
import pickle
import csv
import matplotlib.pyplot as plot

MODEL_PATH = os.getcwd()
MODEL_SRCNAME = 'model.stan'
MODEL_FILENAME = 'model.pkl'
INFERENCE_SRCNAME = 'vectorized_inference_guessprob.stan'
INFERENCE_FILENAME = 'vectorized_inference_guessprob.pkl'
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'Data')  # Parent directory
RESPONSE_NAME = 'responses.csv'
SKILLS_NEEDED_NAME = 'skills_needed.csv'
GUESS_PROB_NAME = 'guess_probs.csv'

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

def divideByScale(guess_probs):
    data_comp = list()
    with open(os.path.join(DATA_PATH, GUESS_PROB_NAME)) as f:
        reader = csv.reader(f)
        ctr = 0
        for row in reader:
            if(ctr==0):
                ctr += 1
                continue
            data_comp.append(row[3])
    div = list()
    for i in range(len(guess_probs)):
        div.append(guess_probs[i]/float(data_comp[i]))
    mean = (sum(div)/len(div))
    return list(map(lambda x:x/mean, guess_probs))

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

def evaluateResponse(legend, response):
    return list(map(lambda x,y:int(x==y), legend, response))

def readResponses():
    convert = {'True':1, 'False':0}
    line_ctr = 0
    is_correct = list()
    ground_truth = list()
    with open(os.path.join(DATA_PATH, RESPONSE_NAME)) as f:
        f_reader = csv.reader(f)
        num_skills = 0
        for data_row in f_reader:
            if(line_ctr==1):
                for value in data_row:
                    if(value==''):
                        num_skills += 1
                legend = data_row[num_skills+1:]
            elif(line_ctr==0):
                line_ctr += 1
                continue
            else:
                is_correct.append(evaluateResponse(legend, data_row[num_skills+1:]))
                temp = list()
            line_ctr += 1
    return is_correct   # 1,0 to indicate correct,wrong for each person

skills_needed = readSkillsNeeded()
is_correct = readResponses()

'''
data_dict = ('isCorrect1', 'isCorrect2', 'isCorrect3')
allowed_values = (0, 1)
observed_data = [dict(zip(data_dict,
                        [(a,b,c) for a in allowed_values for b in allowed_values for c in allowed_values][x])) for x in range(len(allowed_values)**len(data_dict))]
'''

''' ______________________ INFERRING GUESS PROBABILITY FOR EACH QUESTION FROM ALL CANDIDATES ___________________ '''

observed_data = list()
num_questions = len(skills_needed)
num_skills = len(skills_needed[0])
num_candidates = len(is_correct)
#observed_data.append({'num_questions':num_questions, 'num_skills':num_skills, 'num_candidates':num_candidates, 'is_correct':is_correct, 'skillsNeeded':skills_needed})

for qno in range(48):
    spc_correct = list()
    for cand in is_correct:
        spc_correct.append(cand[qno])
    spc_skill = skills_needed[qno]
    observed_data.append({'num_skills':num_skills, 'num_candidates':num_candidates, 'is_correct':spc_correct, 'skillsNeeded':spc_skill})
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

guess_inference = list()

for data in observed_data:
    pass
    result = model.sampling(data=data, pars=('p_guesses'))
    #, pars=('P_skills', 'p_guesses')
    #print(result)
    guess_inference.append(result.summary(pars=('p_guesses'))['summary'][0][0])

print(guess_inference)
guess_inference = divideByScale(guess_inference)
print(guess_inference)
plot.bar([(i+1) for i in range(48)], guess_inference, width = 0.8)
plot.show()
# RESULTS TO BE PLOTTED

''' ______________________________________ INFERENCE USING LEARNED GUESS PROBABILTIES _______________________________ '''

#ground_truth = samples['skills']
is_correct_int = list()
for case in is_correct:
    is_correct_int.append(list(map(lambda x:int(x), case)))

'''
data_dict = ('isCorrect1', 'isCorrect2', 'isCorrect3')
allowed_values = (0, 1)
observed_data = [dict(zip(data_dict,
                        [(a,b,c) for a in allowed_values for b in allowed_values for c in allowed_values][x])) for x in range(len(allowed_values)**len(data_dict))]
'''

observed_data = list()
num_questions = len(skills_needed)
num_skills = len(skills_needed[0])
for case in is_correct_int:
    observed_data.append({'num_questions':num_questions, 'num_skills':num_skills, 'isCorrect':case, 'skillsNeededArr':skills_needed, 'p_guesses':guess_inference})

#Compilation of model
try:
    model_file_path = os.path.join(MODEL_PATH, INFERENCE_FILENAME)
    model_file = open(model_file_path, 'rb')
except FileNotFoundError:
    model_source_path = os.path.join(MODEL_PATH, INFERENCE_SRCNAME)
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
#plotColorMap(ground_truth)
