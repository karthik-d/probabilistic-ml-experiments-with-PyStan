data{
    int<lower=0> N_heads;
    int<lower=0> N_tails;
}

parameters{
    real<lower=0,upper=1> theta;
}

model{
    theta ~ beta(2,2);
    N_heads ~ binomial(N_heads + N_tails, theta);
}
