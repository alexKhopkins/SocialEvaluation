data {
  int<lower=1> nSubjects;                              // number of subjects
  int<lower=1> nTrials;                                // number of trials 
  int<lower=1,upper=2> choice[nTrials,nSubjects];      // 2nd choices, 1 or 2, must be 'int'
  real<lower=-1,upper=1> reward[nTrials,nSubjects];    // outcome, 1 or -1, 'real' is faster than 'int'
  int<lower = 1, upper=2> rs[nTrials,nSubjects]; //realstate
  int<lower = 1, upper=6> state[nTrials,nSubjects]; //state
  real bfne[nSubjects]; //bfne per subject
  
}

parameters {
// DECLARE ALL PARAMETERS AS VECTORS FOR VECTORIZING 
  vector[7] mu_p;
  vector<lower = 0>[7] sigma_p;

 

  //subject-level raw parameters (in unconstrained space)
  vector[nSubjects] spos_raw;
  vector[nSubjects] sneg_raw;
  vector[nSubjects] opos_raw;
  vector[nSubjects] oneg_raw;
  vector[nSubjects] tau_raw;
  vector[nSubjects] init_raw;
  vector[nSubjects] ppos_raw;

}

transformed parameters {

  //subject-level raw parameters
  vector<lower=0, upper=1>[nSubjects] spos;
  vector<lower=0, upper=1>[nSubjects] sneg;
  vector<lower=0, upper=1>[nSubjects] opos;
  vector<lower=0, upper=1>[nSubjects] oneg;
  vector<lower=0, upper=10>[nSubjects] tau;
  vector<lower=-1, upper=1>[nSubjects] init;
  vector<lower=-1, upper=1>[nSubjects] ppos;

  //transform using group level parameters
  for (s in 1:nSubjects) {
    spos[s]  <- Phi_approx( mu_p[1] + sigma_p[1] * spos_raw[s] );
    sneg[s]  <- Phi_approx( mu_p[2] + sigma_p[2] * sneg_raw[s] );
    opos[s]  <- Phi_approx( mu_p[3] + sigma_p[3] * opos_raw[s] );
    oneg[s]  <- Phi_approx( mu_p[4] + sigma_p[4] * oneg_raw[s] );
    tau[s]   <- Phi_approx( mu_p[5] + sigma_p[5] * tau_raw[s] ) * 10;
    init[s] <- -1 + Phi_approx(mu_p[6] + sigma_p[6] * init_raw[s] ) *2;
    ppos[s] <- -1 + Phi_approx(mu_p[7] + sigma_p[7] * ppos_raw[s] ) *2;
  }
}

model {
  // define the value and pe vectors
  vector[2] v[nTrials+1];     // values
  vector[nTrials] pe;         // prediction errors  
  
//priors over group pars
  mu_p ~ normal(0,1);
  sigma_p ~ cauchy(0,5);


// priors over raw (unconstrained) subject level parameters 
  spos_raw ~ normal(0,1);
  sneg_raw ~ normal(0,1);
  opos_raw ~ normal(0,1);
  oneg_raw ~ normal(0,1);
  tau_raw ~ normal(0,1);
  init_raw ~ normal(0,1);
  ppos_raw ~ normal(0,1);
  
  // subject loop and trial loop
  for (s in 1:nSubjects) {

    v[1] <- [init[s],0]';
    
    for (t in 1:nTrials) {
      //* compute action probs using built-in softmax function and related to choice data */
      choice[t,s] ~ categorical_logit( tau[s] * v[t] );
      
      //* prediction error */
      pe[t] <- reward[t,s] - v[t][choice[t,s]];

      //* value updating (learning) */
       if (t == 32 || t == 64 || t == 96 || t == 128 || t == 160) {
          v[t+1] <- [init[s],0]';
       } else {
        if (state[t,s] == 1 || state[t,s] == 2 || state[t,s] == 3) {
          if (rs[t,s] == 1){
      	    v[t+1] <- v[t]; // make a copy of current value into t+1
      	    v[t+1][choice[t,s]] <- v[t][choice[t,s]] + spos[s] * pe[t];
            v[t+1][choice[t,s]] <- v[t+1][choice[t,s]] + ppos[s]; 
           } else {
       	    v[t+1] <- v[t]; // make a copy of current value into t+1
      	    v[t+1][choice[t,s]] <- v[t][choice[t,s]] + sneg[s] * pe[t]; 
           } // end of realstate loop 
         } else {
           if (rs[t,s] == 1){
      	     v[t+1] <- v[t]; // make a copy of current value into t+1
      	     v[t+1][choice[t,s]] <- v[t][choice[t,s]] + opos[s] * pe[t]; 
             v[t+1][choice[t,s]] <- v[t+1][choice[t,s]] + ppos[s]; 
            } else {
       	     v[t+1] <- v[t]; // make a copy of current value into t+1
      	     v[t+1][choice[t,s]] <- v[t][choice[t,s]] + oneg[s] * pe[t]; 
            } // end of realstate loop 

          }//end of state loop

       }// end of trial condition loop

    } // end of loop over trials
  } // end of loop over subjects
} // end of model 

generated quantities {

  // define the value and pe vectors
  vector[2] v2[nTrials+1];     // values
  vector[nTrials] pe2;         // prediction errors  
  real log_lik[nTrials,nSubjects];
  vector[nSubjects] log_lik1;

  // subject loop and trial loop
  for (s in 1:nSubjects) {
    log_lik1[s] <- 0;
    v2[1] <- [init[s],0]';
    
    for (t in 1:nTrials) {
      //* compute action probs using built-in softmax function and related to choice data */
     log_lik1[s] <- log_lik1[s] + categorical_logit_lpmf(choice[t,s] | tau[s] * v2[t] );
     log_lik[t,s] <- categorical_logit_lpmf(choice[t,s] | tau[s] * v2[t] );
      
      //* prediction error */
      pe2[t] <- reward[t,s] - v2[t][choice[t,s]];

      //* value updating (learning) */
       if (t == 32 || t == 64 || t == 96 || t == 128 || t == 160) {
          v2[t+1] <- [init[s],0]';
       } else {
        if (state[t,s] == 1 || state[t,s] == 2 || state[t,s] == 3) {
          if (rs[t,s] == 1){
      	    v2[t+1] <- v2[t]; // make a copy of current value into t+1
      	    v2[t+1][choice[t,s]] <- v2[t][choice[t,s]] + spos[s] * pe2[t]; 
            v2[t+1][choice[t,s]] <- v2[t+1][choice[t,s]] + ppos[s]; 
           } else {
       	    v2[t+1] <- v2[t]; // make a copy of current value into t+1
      	    v2[t+1][choice[t,s]] <- v2[t][choice[t,s]] + sneg[s] * pe2[t]; 
           } // end of realstate loop 
         } else {
           if (rs[t,s] == 1){
      	     v2[t+1] <- v2[t]; // make a copy of current value into t+1
      	     v2[t+1][choice[t,s]] <- v2[t][choice[t,s]] + opos[s] * pe2[t]; 
             v2[t+1][choice[t,s]] <- v2[t+1][choice[t,s]] + ppos[s]; 
            } else {
       	     v2[t+1] <- v2[t]; // make a copy of current value into t+1
      	     v2[t+1][choice[t,s]] <- v2[t][choice[t,s]] + oneg[s] * pe2[t]; 

            } // end of realstate loop 

          }//end of state loop
       }// end of trial condition loop

  } // end of trial loop
 
 }//end of subject loop
}







