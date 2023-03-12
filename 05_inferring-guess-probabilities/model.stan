functions{

    int power(int base, int radical){
        int result = 1;
        for (i in 1:radical){
            result *= base;
        }
        return result;
    }
    
    int countReqdSkills(int[] needed, int ns){
        int ctr = 0;
        for (i in 1:ns){
            if(needed[i]==1){
                ctr += 1;
            }
        }
        return ctr;
    }
}

data{
    int num_candidates;
    int num_skills;
    int<lower=0, upper=1> is_correct[num_candidates];  // Each row is the response evaluation sheet of a candidate
    int<lower=0, upper=1> skillsNeeded[num_skills];
}

transformed data{
    real p_knows = 0.9;
    int num_reqskills = countReqdSkills(skillsNeeded, num_skills);
    int cases = power(2, num_reqskills);
}

parameters{
    real<lower=0, upper=1> p_guesses;
}

model{
    p_guesses ~ beta(2.5,7.5);

    for (i in 1:num_candidates){
        for (n in 1:cases){
            if(n==1){
                is_correct[i] ~ bernoulli(p_knows);
            }
            else{
                is_correct[i] ~ bernoulli(p_guesses);
            }
        }
    }
}
