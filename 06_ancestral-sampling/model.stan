functions{
    int[] sampleCorrects_rng(int nq, int ns, int[] skill_set, matrix needed, real p_knows, real p_guesses){

        row_vector[nq] exist_sum;
        row_vector[nq] req_neg_sum;   // Negative of number of more skills required to pass for a question
        row_vector[ns] skill_set_local;
        real attempt_prob;
        int isCorrect[nq];

        for (i in 1:ns){
            skill_set_local[i] = skill_set[i];
        }

        exist_sum = skill_set_local*(needed');
        req_neg_sum = exist_sum;
        for (i in 1:ns){
            req_neg_sum -= needed'[i,];
        }

        for (i in 1:nq){
            attempt_prob = req_neg_sum[i]==0 ? p_knows : p_guesses;
            isCorrect[i] = bernoulli_rng(attempt_prob);
            print(isCorrect[i], attempt_prob);
        }
        return isCorrect;
    }
}

data{
    int num_skills;
    int num_questions;
    matrix<lower=0,upper=1>[num_questions, num_skills] skillsNeeded;  // Each row corresponds to skills for a particular question
}

generated quantities{
    int<lower=0,upper=1> skills[num_skills];   // Each sample is like one case of sample combination
    int<lower=0,upper=1> isCorrect[num_questions];

    // Ancestral Sampling for the sampled skills
    for (i in 1:num_skills){
        skills[i] = bernoulli_rng(0.5);  // Random Number Generator
    }

    // Deducing 'ideal-case' values of isCorrect(s) per the model
    isCorrect = sampleCorrects_rng(num_questions, num_skills, skills, skillsNeeded, 0.9, 0.2);
}
