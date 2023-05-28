# Probabilistic Machine Learning Experiments with PyStan

Probabilistic modeling using PyStan with demonstrative case study experiments from Christopher Bishop's Model-based Machine Learning.

> **Note that** the the first run of all `*/infer.py` files will be slow since the PM model will be built and stored as `pickle` files. Subsequent runs will reuse this `pickle` file.

> **Be sure to** remove or relocate the corresponding `*.pkl` file(s) when changing model configurations. The older model will be used for inference, otherwise.

## Experiment: Mapping MCQ Test Responses to Candidate Skills

> An elaborate case study description can be found in [MBML Book, Chapter 2](https://www.mbmlbook.com/LearningSkills.html).

- Candidates take a multiple-choice test comprising 5 choices per question, with exactly one right answer per question.
- Each question is associated with a set of skills (one or more), that forms a part of the given dataset.
- **Goal**: Determine which skills each candidate has, and with what probability, given their answers in the test. 
- **Dataset**: Skill ground truth and response data for **22 candidates**, across **48 questions**, to assess **7 skills**, is contained in CSV files in the [data](./data) directory.

The following binary heatmap represents the skills, one or more, assessed by each of the 48 questions in the dataset.
<img src="./assets/skill-question-map.png" alt="skill-question-map" />


### Incremental Solution Implementation

The solution is implemented as a probabilstic model that makes the following initial set of assumptions on the data,
1. A candidate has either mastered each skill or not.
2. Before seeing any test results, it is equally likely that a candidate does or doesn’t have any particular skill.
3. If a candidate has all of the skills needed for a question, then they are likely to make a mistake once in ten times (a 90% right answer probability).
4. If a candidate doesn’t have all the skills needed for a question, they will pick an answer at random. Hence, there’s a one in five chance that they get the question right (a 20% right answer probability, assuming a uniform guessing distribution of course!).
5. Whether the candidate gets a question right depends only on what skills that candidate has, and not on anything else. 

Assumptions 3 and 4 essentially give rise to a **model parameter** each, and they can be fine-tuned over time.

The following **factor graph** represents the model, its assumptions, and data flow.

#### Non-Vectorized Primitive Models

- The non-vectorized models are primitive implementations based on small subsets of data.
- They capture all possible candidate response combinations.
- They provide a way to intuitively ensure that the model is foundationally right, and that the assumptions and inference workflows are valid. 

##### Three-Question Model

- [Link to three-question model implementation](./03_three-question-model).
- Modeled for **2 skills** assessed through **3 questions****.
- These are **skills 1 and 7**; and **questions 1 through 3 on the skill-question heatmap. 
- Factors and evaluates skill probabilities for all possible response combinations.

##### Four-Question Model

- [Link to four-question model implementation](./04_four-question-model).
- Modeled for **2 skills** assessed through **4 questions**.
- These are **skills 1 and 7**; and **questions 1 through 3 on the skill-question heatmap. 
- Factors and evaluates skill probabilities for all possible response combinations.

#### Complete Vectorized Model

- [Link to model implementation on complete dataset](./05_vectorized-model).
- This is the first realisitic models that uses the complete dataset for inference.
- It carries the original model assumptions.

THe following three-feature heatmap represents the correct and incorrect reponses of the 22 candidates to the 48 questions.   
- White blocks represent questions answered correctly, where colored boxes represent incorrect responses.
- The colors also mark each incorrect response with the skills required to answer them.
- The heatmap helps visually and qualitatively assess, which candidate likely lacks what skills.

The inferred skill probabilities are compared against the ground truth data on skills possessed by each of the 22 candidates in the binary heatmap below.

### Model Improvement: Learning Guess Probabilities

- To improve the probabilistic model, the guess probabilities are no longer assumed to be constant.
- Rather, the guess probability for each question is inferred through **message passing** and **belief propagation**  in the undirected factor graph.
- Specificially, the attempts of each of the 22 candidates for the 48 questions and learning the guess probabilities from applicable scenarios i.e when the candidate does not possess all
the skills required to answer a question.   

**Guess Probability**: Probability that a candidate gets an answer right by guessing, when they do not have the skills necessary to answer the
question.

   
It is worth mentioning that the ground truth from the dataset was not used. Rather, all possible combinations of skill-sets were generated and used to infer the
posterior for guess probabilities.

A substantial improvement in the inference is evident after learning the guess probabilities.      
Further improvements can be made, for instance, by learning the “**know probabilities**".

## Other Experiments

## References

- [Model-Based Machine Learning *by John Winn*](https://www.mbmlbook.com/).
- [Equivalent C# Infer.NET Implementation](https://github.com/dotnet/mbmlbook/tree/main).
- [PySTAN Documentation](https://pystan.readthedocs.io/en/latest/).
- [STAN Documentation](https://mc-stan.org/users/documentation/).
