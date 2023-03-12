import os
import pystan
import pickle
import csv
import matplotlib.pyplot as plot

MODEL_PATH = os.getcwd()
MODEL_SRCNAME = 'model.stan'
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
                for i in range(num_skills):
                    temp.append(convert[data_row[1:1+num_skills][i]])
                ground_truth.append(temp)
            line_ctr += 1
    print(ground_truth)
    return is_correct, ground_truth   # 1,0 to indicate correct,wrong for each person

skills_needed = readSkillsNeeded()
is_correct, ground_truth = readResponses()

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

inference = list()   # Matrix of skill_probabilities for each observed_data

for data in observed_data:
    result = model.sampling(data=data, pars=('P_skills'))
    curr = list()
    for k in range(num_skills):
        curr.append(result.summary(pars=('P_skills'))['summary'][k][0])
    inference.append(curr)            # Mean Probabilties of All Skills

    '''
    summary = result.summary(pars=('P_skill1', 'P_skill2'))
    inference['skill1'].append(summary['summary'][0][0])
    inference['skill2'].append(summary['summary'][1][0])

    plot.hist(result['skill1'], bins=[x*0.01 for x in range(0,101)], rwidth=0.9)
    plot.title('Probability Distribution')
    plot.show()

    plot.hist(result['skill2'], bins=[x*0.01 for x in range(0,101)], rwidth=0.9)
    plot.title('Probability Distribution')
    plot.show()

    plot.hist(result['hasSkills'], bins=[x*0.01 for x in range(0,101)], rwidth=0.9)
    plot.title('Probability Distribution')
    plot.show()
    '''

plotColorMap(inference)
plotColorMap(ground_truth)
# RESULTS TO BE PLOTTED
