functions{

    int power(int base, int radical){
        int result = 1;
        for (i in 1:radical){
            result *= base;
        }
        return result;
    }

    /*matrix getCombinationMatrix(int cols_){
        int rows_ = 2*cols_;
        matrix[rows_, cols_] comb_matrix;
        row_vector[cols_] flags;
        for (i in 1:rows_){
            comb_matrix = append_row(comb_matrix, flags);
            for (j in 1:cols_){
                if(i%(power(2, cols_-j))==0){  // Toggle 0 and 1
                    if(flags[j]==0){
                        flags[j] = 1;
                    }
                    else{
                        flags[j] = 0;
                    }
                    //flags[j] = (flags[j]+1)%2; // DOESN'T work SINCE flags IS REAL(VECTOR)
                }
            }
        }
        return comb_matrix;
    }*/

    int[,] getCombinationMatrix(int cols_){
        int rows_ = power(2, cols_);
        int comb_matrix[rows_, cols_];
        int flags[cols_];   // Doesn't initialize to 0 on its own, though the docs mention this!
        for (i in 1:cols_){
            flags[i] = 0;
        }
        for (i in 1:rows_){
            comb_matrix[i] = flags;
            for (j in 1:cols_){
                if(i%(power(2, cols_-j))==0){  // Toggle 0 and 1
                    flags[j] = (flags[j]+1)%2; // DOESN'T work IF flags IS REAL(VECTOR)
                }
            }
        }
        return comb_matrix;
    }

    matrix getAttemptProbabilties(int nq, int ns, int[,] skillset_cases, matrix needed, real p_knows, real p_guesses){
        int cases = power(2, ns);
        matrix[cases, nq] exist_sum;
        matrix[cases, nq] req_neg_sum;   // Negative of number of more skills required to pass for a question
        matrix[cases, nq] attempt_prob;   // Either knows or guesses
        matrix[cases, ns] skillset_cases_local;

        for (i in 1:power(2, ns)){
            for (j in 1:ns){
                skillset_cases_local[i,j] = skillset_cases[i,j];
            }
        }

        exist_sum = skillset_cases_local*(needed');
        req_neg_sum = exist_sum;
        for (i in 1:cases){
            for (j in 1:ns){
                req_neg_sum[i,] -= needed'[j,];
            }
        }

        for (i in 1:cases){
            for (j in 1:nq){
                attempt_prob[i,j] = (req_neg_sum[i,j]==0 ? p_knows : p_guesses);
            }
        }

        return attempt_prob;
    }

    vector evaluateLikelihoods(int nq, int ns, int[,] skillset_cases, matrix needed, real p_knows, real p_guesses, int[] isCorrect){
        int cases = power(2, ns);
        matrix[cases, nq] attempt_prob;   // Either knows or guesses
        vector[cases] likelihoods;

        attempt_prob = getAttemptProbabilties(nq, ns, skillset_cases, needed, p_knows, p_guesses);

        for (i in 1:cases){
            likelihoods[i] = 0;
            for (j in 1:nq){
                likelihoods[i] += bernoulli_lpmf(isCorrect[j] | attempt_prob[i,j]);
            }
        }
        return likelihoods;
    }
}

data{
    int num_questions;
    int num_skills;
    int<lower=0,upper=1> isCorrect[num_questions] ;
    int<lower=0,upper=1> skillsNeededArr[num_questions, num_skills];  // Each row corresponds to skills for a particular question
}

transformed data{
    int cases = power(2,num_skills);
    //int<lower=0,upper=1> skillset_cases[power(2, num_skills), num_skills];
    int skillset_cases[cases, num_skills];
    matrix<lower=0,upper=1>[num_questions, num_skills] skillsNeeded;
    real<lower=0,upper=1> p_knows;
    real<lower=0,upper=1> p_guesses;
    vector[cases] likelihoods;
    //matrix<lower=0,upper=1>[power(2, num_skills), num_skills] skillset_cases;
    for (i in 1:num_questions){
        for (j in 1:num_skills){
            skillsNeeded[i,j] = skillsNeededArr[i,j];
        }
    }
    skillset_cases = getCombinationMatrix(num_skills);

    p_knows = 0.9;
    p_guesses = 0.2;
    likelihoods = evaluateLikelihoods(num_questions, num_skills, skillset_cases, skillsNeeded, p_knows, p_guesses, isCorrect);
}

parameters{
    row_vector<lower=0,upper=1>[num_skills] skills;
}

model{
    skills ~ beta(2,2);

    for(i in 1:cases){
        target += bernoulli_lpmf(skillset_cases[i,]|skills);
        target += likelihoods[i];
    }
}

generated quantities{
    vector[num_skills] P_skills;
    vector[cases] priors;

    for (i in 1:num_skills){
        P_skills[i] = log(0);
        for (j in 1:cases){
            if(skillset_cases[j,i]==1){
                P_skills[i] = log_sum_exp(P_skills[i], bernoulli_lpmf(skillset_cases[j,]|skills)+likelihoods[j]);
            }
            if(i==1){
                // ONLY EXECUTED ONCE PER 'j'
                priors[j] = bernoulli_lpmf(skillset_cases[j,]|skills) + likelihoods[j];
            }
        }
    }
    P_skills = exp(P_skills - log_sum_exp(priors));
}
