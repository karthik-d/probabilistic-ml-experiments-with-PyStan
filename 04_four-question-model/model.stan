data{
    int<lower=0,upper=1> isCorrect1;
    int<lower=0,upper=1> isCorrect2;
    int<lower=0,upper=1> isCorrect3;
    int<lower=0,upper=1> isCorrect4;
}

parameters{
    real<lower=0,upper=1> skill1;
    real<lower=0,upper=1> skill2;
}

transformed parameters{
    real<lower=0,upper=1> p_knows;
    real<lower=0,upper=1> p_guesses;
    vector[4] likelihoods;

    p_knows = 0.9;
    p_guesses = 0.2;
    likelihoods[1] = bernoulli_lpmf(isCorrect1|p_knows)+bernoulli_lpmf(isCorrect2|p_knows)+bernoulli_lpmf(isCorrect3|p_knows)+bernoulli_lpmf(isCorrect4|p_knows);
    likelihoods[2] = bernoulli_lpmf(isCorrect1|p_knows)+bernoulli_lpmf(isCorrect2|p_guesses)+bernoulli_lpmf(isCorrect3|p_guesses)+bernoulli_lpmf(isCorrect4|p_guesses);
    likelihoods[3] = bernoulli_lpmf(isCorrect1|p_guesses)+bernoulli_lpmf(isCorrect2|p_knows)+bernoulli_lpmf(isCorrect3|p_guesses)+bernoulli_lpmf(isCorrect4|p_guesses);
    likelihoods[4] = bernoulli_lpmf(isCorrect1|p_guesses)+bernoulli_lpmf(isCorrect2|p_guesses)+bernoulli_lpmf(isCorrect3|p_guesses)+bernoulli_lpmf(isCorrect4|p_guesses);
}

model{
    skill1 ~ beta(2,2);
    skill2 ~ beta(2,2);

    target += log(skill1) + log(skill2) + likelihoods[1];
    target += log(skill1) + log(1-skill2) + likelihoods[2];
    target += log(1-skill1) + log(skill2) + likelihoods[3];
    target += log(1-skill1) + log(1-skill2) + likelihoods[4];
}

generated quantities{
    real P_skill1;
    real P_skill2;

    P_skill1 = log_sum_exp(log(skill1)+log(skill2)+likelihoods[1], log(skill1)+log(1-skill2)+likelihoods[2]);
    P_skill1 -= log_sum_exp(P_skill1, log_sum_exp(log(1-skill1)+log(skill2)+likelihoods[3], log(1-skill1)+log(1-skill2)+likelihoods[4]));
    P_skill1 = exp(P_skill1);

    P_skill2 = log_sum_exp(log(skill1)+log(skill2)+likelihoods[1], log(1-skill1)+log(skill2)+likelihoods[3]);
    P_skill2 -= log_sum_exp(P_skill2, log_sum_exp(log(skill1)+log(1-skill2)+likelihoods[2], log(1-skill1)+log(1-skill2)+likelihoods[4]));
    P_skill2 = exp(P_skill2);
}
