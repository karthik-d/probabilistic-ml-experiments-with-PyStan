# Probabilistic Machine Learning Experiments with PyStan

Probabilistic modeling using PyStan with demonstrative case study experiments from Christopher Bishop's Model-based Machine Learning.

> **Note that** the the first run of all `*/infer.py` files will be slow since the PM model will be built and stored as `pickle` files. Subsequent runs will reuse this `pickle` file.

> **Be sure to** remove or relocate the corresponding `*.pkl` file(s) when changing model configurations. The older model will be used for inference, otherwise.

## Experiment: Mapping MCQ Test Responses to Candidate Skills

> An elaborate case study description can be found in [MBML Book, Chapter 2](https://www.mbmlbook.com/LearningSkills.html).

- Candidates take a multiple-choice test comprising 5 choices per question, with exactly one right answer per question.
- Each question is associated with a set of skills (one or more), that forms a part of the given dataset.
- **Goal**: Determine which skills each candidate has, and with what probability, given their answers in the test. 
- **Dataset**: Skill ground truth and response data for **22 candidates**, across **48 questions**, to assess **7 skills**, is contained in CSV files in the [data directory](./data).

The following binary heatmap represents the skills, one or more, assessed by each of the 48 questions in the dataset.   
   
<img src="./assets/skill-question-map.png" alt="skill-question-map" />


### Incremental Solution Implementation

The solution is implemented as a probabilstic model that makes the following initial set of assumptions on the data,

<ol type="A"><i>
<li>A candidate has either mastered each skill or not.</li>
<li>Before seeing any test results, it is equally likely that a candidate does or doesn’t have any particular skill.</li>
<li>If a candidate has all of the skills needed for a question, then they are likely to make a mistake once in ten times -- a 90% right answer probability.</li>
<li>If a candidate doesn’t have all the skills needed for a question, they will pick an answer at random. Hence, there’s a one in five chance that they get the question right -- a 20% right answer probability, assuming a uniform guessing distribution of course!</li>
<li>Whether the candidate gets a question right depends only on what skills that candidate has, and not on anything else.</li>
</i></ol>
   
_Assumptions C and D_ essentially give rise to a **model parameter** each, and they can be fine-tuned over time.

#### Non-Vectorized Primitive Models

- The non-vectorized models are primitive implementations based on small subsets of data.
- They capture all possible candidate response combinations.
- They provide a way to intuitively ensure that the model is foundationally right, and that the assumptions and inference workflows are valid. 

##### Three-Question Model

- [Link to three-question model implementation](./03_three-question-model).
- Modeled for **2 skills** assessed through **3 questions****.
- These are **skills 1 and 7**; and **questions 1 through 3 on the skill-question heatmap. 
- Factors and evaluates skill probabilities for all possible response combinations.

The following **factor graph** represents the model and message flow for the three-question scenario. 
   
<img src="./assets/3q-factor-graph.png" alt="3q-factor-graph" />

##### Four-Question Model

- [Link to four-question model implementation](./04_four-question-model).
- Modeled for **2 skills** assessed through **4 questions**.
- These are **skills 1 and 7**; and **questions 1 through 3 on the skill-question heatmap. 
- Factors and evaluates skill probabilities for all possible response combinations.

The following **factor graph** represents the model and message flow for the four-question scenario.  
   
<img src="./assets/4q-factor-graph.png" alt="4q-factor-graph" />

#### Baseline Vectorized Model

- [Link to model implementation on complete dataset](./05_vectorized-model).
- This is the first realisitic models that uses the complete dataset for inference.
- It carries the original model assumptions.
- The implementation used matrix operations for message passing and inference to manage larger datasets effectively, and to optimize for a GPU.

THe following three-feature heatmap represents the correct and incorrect reponses of the 22 candidates to the 48 questions.   
- White blocks represent questions answered correctly, where colored boxes represent incorrect responses.
- The colors also mark each incorrect response with the skills required to answer them.
- The heatmap helps visually and qualitatively assess, which candidate likely lacks what skills.

The inferred skill probabilities are compared against the ground truth data on skills possessed by each of the 22 candidates in the binary heatmap below.   
   
<img src="./assets/result-baseline.png" alt="result-baseline" width="300" />

#### Improved Vectorized Model: Learning Guess Probabilities

- To improve the probabilistic model, the guess probabilities are no longer assumed to be constant.
- Instead, the guess probability for each question is inferred through **message passing** and **belief propagation**  in the undirected factor graph.
- Specificially, assumption D is modified as follows,

<i><q>If a candidate doesn’t have all the skills needed for a question, they will pick an answer at random. **The probability of getting a question right, called the guess probability, is inferred from data**.</q></i> 

The inferred skill probabilities, when applying learnt guess probabilities, are compared against baseline performance and the ground truth data on skills possessed by each of the 22 candidates in the binary heatmap below.   
   
<img src="./assets/result-improved.png" alt="result-improved" width="500" />
   
> Note that the ground truth is not used for this inference.    
> Rather, all possible combinations of 'skill-sets' are generated and 'belief propagated' to infer the posterior for guess probabilities.

A substantial improvement in the inference is evident after learning the guess probabilities.      
Further improvements can be made, for instance, by learning the “**know probabilities**".

## Other Experiments

## References

- [Model-Based Machine Learning *by John Winn*](https://www.mbmlbook.com/).
- [Equivalent C# Infer.NET Implementation](https://github.com/dotnet/mbmlbook/tree/main).
- [PySTAN Documentation](https://pystan.readthedocs.io/en/latest/).
- [STAN Documentation](https://mc-stan.org/users/documentation/).
