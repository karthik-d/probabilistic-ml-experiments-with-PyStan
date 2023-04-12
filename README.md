# Probabilistic Machine Learning Experiments with PyStan

Probabilistic modeling using PyStan with demonstrative case study experiments from Christopher Bishop's Model-based Machine Learning.

> **Note that** the the first run of all `*/infer.py` files will be slow since the PM model will be built and stored as `pickle` files. Subsequent runs will reuse this `pickle` file.

> **Be sure to** remove or relocate the corresponding `*.pkl` file(s) when changing model configurations. The older model will be used for inference, otherwise.

## Experiment: Mapping MCQ Test Responses to Candidate Skills

<add description>

### Model Improvement: Learning Guess Probabilities.

In an attempt to improve the model, it is made modeled without assuming constant guess probabilities. Since each question is
likely to have a different difficulty level, the guess probabilities for each question are inferred. This is done using the attempts 
of each of the 22 candidates for the 48 questions and learning the guess probabilities from applicable scenarios i.e when the candidate does not possess all
the skills required to answer a question.   

**Guess Probability**: Probability that a candidate gets an answer right by guessing, when they do not have the skills necessary to answer the
question.

   
It is worth mentioning that the ground truth from the dataset was not used. Rather, all possible combinations of skill-sets were generated and used to infer the
posterior for guess probabilities.

A substantial improvement in the inference is evident after learning the guess probabilities.      
Further improvements can be made, for instance, by learning the “**know probabilities**".

## Other Experiments

## References

- [PySTAN Documentation](https://pystan.readthedocs.io/en/latest/)
- [STAN Documentation](https://mc-stan.org/users/documentation/)
