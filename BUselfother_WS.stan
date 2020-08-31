functions {
  real betall(vector a, vector r, vector rs, vector state, real alphaself, real betaself, real alphaother, real betaother, real memself, real memother, real Tau, real alphainit, real betainit){
   
    // Declare & initialize
    real  sll = 0.5;                   // sum log lik over trials
    int   nTrials = 192;  
    int   qTrials = 193;   
    real  policy;
    matrix[qTrials, 2] q;
    vector[qTrials] alphatask;
    vector[qTrials] betatask;
    matrix[nTrials, 2] pol;

    
    for (trN in 1:nTrials){

    if (trN == 1) {
     if (state[trN] == 1 || state[trN] == 2 || state[trN] == 3) {
        alphatask[1] <- 1;
        betatask[1] <- 1;
        q[trN,1] <- alphainit + alphaself;
        q[trN,2] <- betainit + betaself;
     } else {
        alphatask[1] <- 1;
        betatask[1] <- 1;
        q[trN,1] <- alphainit + alphaother;
        q[trN,2] <- betainit + betaother;
     } //end of state initial trial loop
   }//end of first trial loop


      // probability of choosing action 1:
      pol[trN,1] = 1/(1+exp((q[trN,2]-q[trN,1])/(Tau*(q[trN,2]+q[trN,1]))));
      pol[trN,2] = 1-pol[trN,1];

      policy = pol[trN,1];
      
      // increment of log-likelihood:
      if (a[trN] == 1){
       sll += log(policy);
      } else {
       sll += log(1-policy); 
      }

      

      // finally, learning:
	if (state[trN] == 1 || state[trN] == 2 || state[trN] == 3) {
         if (trN == 32 || trN == 64 || trN == 96 || trN == 128 || trN == 160) {
           alphatask[trN+1] <- 1;
           betatask[trN+1] <- 1;
           //alphastart <- alphaself + alphatask[trN+1];
           //betastart <- betaself + betatask[trN+1];
           q[trN+1,1] <- alphaself + alphainit;
           q[trN+1,2] <- betaself + betainit;     
         } else {
      	  alphatask[trN+1] <- (1-memself)*alphatask[trN] + memself + rs[trN]; ; 
      	  betatask[trN+1] <- (1-memself)*betatask[trN] + memself + (1-rs[trN]);  
          q[trN+1,1] <- alphaself + alphatask[trN+1];
	  q[trN+1,2] <- betaself + betatask[trN+1];
         }
        } else {  
         if (trN == 32 || trN == 64 || trN == 96 || trN == 128 || trN == 160) {
           alphatask[trN+1] <- 1;
           betatask[trN+1] <- 1;
           //alphastart <- alphaother + alphatask[trN+1];
           //betastart <- betaother + betatask[trN+1];
           q[trN+1,1] <- alphaother + alphainit;
           q[trN+1,2] <- betaother + betainit;
         } else { 
      	  alphatask[trN+1] <- (1-memother)*alphatask[trN] + memother + rs[trN]; ; 
      	  betatask[trN+1] <- (1-memother)*betatask[trN] + memother + (1-rs[trN]);  
          q[trN+1,1] <- alphaother + alphatask[trN+1];
	  q[trN+1,2] <- betaother + betatask[trN+1];
         } // end of initial trial loop
      } // end of state loop
   

    } // end loop over trials 
    
    return sll; 
  }    // end of function rw1lr_log
  
}


data {
  int<lower=1> totPtN;                           // number of subjects
  int nstates;
  int nactions;
  int<lower=1> nTrials; 
  matrix<lower=1, upper=6>[nTrials,totPtN] state;   // current state
  matrix<lower=1,upper=2>[nTrials,totPtN]  action;  // 1st choices, 1 or 2
  matrix<lower=-1,upper=1>[nTrials,totPtN] reward;    // outcome, 1 or -1
  matrix<lower = 0, upper=1> [nTrials,totPtN] rs; //realstate //1 is positive
  real bfne[totPtN]; //bfne per subject
}

parameters {

// DECLARE ALL PARAMETERS AS VECTORS FOR VECTORIZING 
  vector[9] mu_p;
  vector<lower = 0>[9] sigma_p;


  //subject-level raw parameters
  vector[totPtN] alphaself_raw;
  vector[totPtN] alphaother_raw;
  vector[totPtN] betaself_raw;
  vector[totPtN] betaother_raw;
  vector[totPtN] memself_raw;
  vector[totPtN] memother_raw;
  vector[totPtN] Tau_raw;
  vector[totPtN] alphainit_raw;
  vector[totPtN] betainit_raw;

}

transformed parameters {
  
  //constrained parameters
  vector<lower=0, upper=200>[totPtN] alphaself;
  vector<lower=0, upper=200>[totPtN] alphaother;
  vector<lower=0, upper=200>[totPtN] betaself;
  vector<lower=0, upper=200>[totPtN] betaother;
  vector<lower=0, upper=1>[totPtN] memself;
  vector<lower=0, upper=1>[totPtN] memother;
  vector<lower=0, upper=15>[totPtN] Tau;
  vector<lower=0, upper=200>[totPtN] alphainit;
  vector<lower=0, upper=200>[totPtN] betainit;

  for (s in 1:totPtN) {

    alphaself[s]  <- Phi_approx( mu_p[1] + sigma_p[1] * alphaself_raw[s] ) *50;
    alphaother[s]  <- Phi_approx( mu_p[2] + sigma_p[2] * alphaother_raw[s] ) *50;
    betaself[s]  <- Phi_approx( mu_p[3] + sigma_p[3] * betaself_raw[s] )*50;
    betaother[s]  <- Phi_approx( mu_p[4] + sigma_p[4] * betaother_raw[s] )*50;
    memself[s] <- Phi_approx(mu_p[5] + sigma_p[5] * memself_raw[s] );
    memother[s] <- Phi_approx(mu_p[6] + sigma_p[6] * memother_raw[s] );
    Tau[s] <- 10*Phi_approx( mu_p[7] + sigma_p[7]* Tau_raw[s] )+ 0.2;
    alphainit[s] <- 50*Phi_approx( mu_p[8] + sigma_p[8] * alphainit_raw[s] );
    betainit[s] <- 50*Phi_approx( mu_p[9] + sigma_p[9] * betainit_raw[s] );
  }

}

model {
  
//priors over group mus
  mu_p ~ normal(0,1);
  sigma_p ~ cauchy(0,5);



  //priors over raw (unconstrained)subject level parameters
  alphaself_raw ~ normal(0,1);
  alphaother_raw ~ normal(0,1);
  betaself_raw  ~ normal(0,1);
  betaother_raw  ~ normal(0,1);
  memself_raw   ~ normal(0,1);
  memother_raw ~ normal(0,1);
  Tau_raw   ~ normal(0,1);
  alphainit_raw ~ normal(0,1);
  betainit_raw ~ normal(0,1);


	for (ptN in 1:totPtN){
	 target += betall(action[1:nTrials,ptN], reward[1:nTrials,ptN], rs[1:nTrials,ptN], state[1:nTrials,ptN], alphaself[ptN], betaself[ptN], alphaother[ptN], betaother[ptN], memself[ptN], memother[ptN], Tau[ptN], alphainit[ptN], betainit[ptN]);
	} //end of participant loop
	  
}  // end of model



generated quantities {

  vector[totPtN] log_lik1;

	for (ptN in 1:totPtN){
	 log_lik1[ptN] = betall(action[1:nTrials,ptN], reward[1:nTrials,ptN], rs[1:nTrials,ptN], state[1:nTrials,ptN], alphaself[ptN], betaself[ptN], alphaother[ptN], betaother[ptN], memself[ptN], memother[ptN], Tau[ptN], alphainit[ptN], betainit[ptN]);
	} //end of participant loop
	  
} //end of GQ block



