# Probabilistic Machine Learning Experiments with PyStan

Probabilistic modeling using PyStan with demonstrative case study experiments from Christopher Bishop's Model-based Machine Learning.

> **Note that** the the first run of all `*/infer.py` files will be slow since the PM model will be built and stored as `pickle` files. Subsequent runs will reuse this `pickle` file.

> **Be sure to** remove or relocate the corresponding `*.pkl` file(s) when changing model configurations. The older model will be used for inference, otherwise.

## Experiment: Mapping MCQ Test Responses to Candidate Skills

<add description>

### Model Improvement: Learning Guess Probabilities.

In an attempt to improve the model, it is made more realistic by assuming that each question is
bound to have its own difficulty level, the guess probabilities for each question are learnt. This is
done using the attempts of each of the 22 candidates for the 48 questions and learning the
guess probabilities from the scenarios where it is applicable i.e when he does not possess all
the skills needed to answer the questions. It may be noted that the ground truth from the dataset
was not used. Instead, all possible cases of the skill sets were generated and used to infer the
posterior for guess probability

A substantial improvement in the inference is evident after learning the guess probabilities.      
Further improvements can be made, for instance, by learning the “**know probabilities**".

## Other Experiments

## References

- [PySTAN Documentation](https://pystan.readthedocs.io/en/latest/)
- [STAN Documentation](https://mc-stan.org/users/documentation/)
