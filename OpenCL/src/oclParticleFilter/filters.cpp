#include <math.h>
#include "filter_utils.h"

/**
   This file contains the function parameters for model parameter estimators 
   as defined in Alireza Javaheri's book, "Inside Volatility Arbitrage"
**/

/*
  log_stock_prices are the log of stock prices
  muS is the real-world stock drift
  n_stock_prices is the number of the above stock prices
  (omega, theta, xi, rho) are the Heston parameters
  u[] is the set of means of observation errors
  v[] is the set of variances of observation errors
  estimates[] are the estimated observations from the filter
*/
// Having u[] and v[] we can evaluate the (minus log) Likelihood as
// the sum of log(v[i1])+u[i1]*u[i1]/v[i1]
// and minimize the sum in order to obtain the optimal parameters
// the minimization could be done for instance via the direction set
// routine available in the Numerical Recipes in C
void estimate_ekf_parm_1_dim_heston (
                                                double *log_stock_prices,
                                                double muS,
                                                int n_stock_prices,
                                                double omega,
                                                double theta,
                                                double xi,
                                                double rho,
                                                double *u,
                                                double *v,
                                                double *estimates)
{
  int i1;
  double x, x1, W, H, A;
  double P, P1, z, U, K;
  const double delt=1.0/252.0;
  const double eps=0.00001;
  x = 0.04;
  P=0.01;
  u[0]=u[n_stock_prices-1]=0.0;
  v[0]=v[n_stock_prices-1]=1.0;
  estimates[0]=estimates[1]=log_stock_prices[0]+eps;

  for (i1=1;i1<n_stock_prices-1;i1++)
	{
      if (x<0) 
        x=0.00001;
		
	  //2.15. TODO: Replace with 2.27 pg 121.
      x1 = x + (omega-rho*xi*muS - (theta-0.5*rho*xi) * x) * delt 
			 + rho * xi * (log_stock_prices[i1]-log_stock_prices[i1-1]);
      //after 2.16. TODO: replace with 2.27, pg 121
	  A = 1.0-(theta-0.5*rho*xi)*delt;
      W = xi*sqrt((1-rho*rho) * x * delt); //TODO: Update with page 121
      
	  //From the Generic EKF algo. 
	  P1 = W*W + A*P*A;
      if (x1<0) 
        x1=0.00001;

      H = -0.5*delt; //after 2.16. Ok asis
      U = sqrt(x1*delt); //after 2.16, in 2.15, x1 in code is. OK as is.
	  //is the same as x_k in the 2.5. Here it becomes v_k

      K = P1*H/( H*P1*H + U*U); //KF: Kalman Gain 2.11

      z = log_stock_prices[i1+1]; //next actual price

      x = x1 + K * (z - (log_stock_prices[i1] + (muS-0.5*x1)*delt)); //KF: measurement updates 2.9
      u[i1] = z - (log_stock_prices[i1] + (muS-0.5*x1)*delt); //means of observation errors (MPE?)
      v[i1] = H*P1*H + U*U; //variances of observation errors
      estimates[i1+1] = log_stock_prices[i1] + (muS-0.5*x1)*delt; //next estimate
      P=(1.0-K*H)*P1; //KF: P update
	}
}


/*
  log_stock_prices are the log of stock prices
  muS is the real-world stock drift
  n_stock_prices is the number of the above stock prices
  (omega, theta, xi, rho) are the Heston parameters
  u[] is the set of means of observation errors
  v[] is the set of variances of observation errors
  estimates[] are the estimated observations from the filter
*/
void estimate_ekf_parm_1_dim (
	double *log_stock_prices,
	double muS,
	int n_stock_prices,
	double omega,
	double theta,
	double xi,
	double rho,
	double p,
	double *u,
	double *v,
	double *estimates)
{
  int i1;
  double x, x1, W, H, A;
  double P, P1, z, U, K;
  const double delt=1.0/252.0;
  const double eps=0.00001;
  x = 0.04; //initialized to 0.04 i.e. 40% as specified in the problem
  P=0.01;
  u[0]=u[n_stock_prices-1]=0.0;
  v[0]=v[n_stock_prices-1]=1.0;
  estimates[0]=estimates[1]=log_stock_prices[0]+eps;

  for (i1=1;i1<n_stock_prices-1;i1++)
	{
      if (x<0) 
        x=0.00001;
	  
	  /*
	  Eqn 2.27 pg 121
	  v[k] =  
		+ [ omega - rho * xi * muS * v[k-1]^(p - 0.5) - (theta - 0.5 * rho * xi * v[k-1]^(p - 0.5)) * v[k-1] ] * delta_t
		+ roe * xi * v[k-1]^(p - 0.5) * ln(S[k]/S[K-1]) 
	  */
      x1 = x 
		+ (omega - rho*xi*muS*pow(x, p - 0.5) - (theta-0.5*rho*xi*pow(x, p - 0.5)) * x) * delt 
		+ rho * xi * pow(x, p - 0.5) * (log_stock_prices[i1]-log_stock_prices[i1-1]);
     //Eqn on page 121
	  A = 1.0 
		  - ((rho * xi * muS) * (p - 0.5) * pow(x, p - 1.5) + theta - 0.5 * rho * (p + 0.5) * pow(x, p - 0.5)) * delt
		  + (p - 0.5) * rho * xi * pow(x, p - 1.5) * (log_stock_prices[i1]-log_stock_prices[i1-1]);
	  //Eqn on page 121
      W = xi * sqrt((1 - rho * rho) * delt) * pow(x, p);

	  P1 = W*W + A*P*A;
      if (x1<0) 
        x1=0.00001;

      H = -0.5 * delt;
	  U = sqrt(x1 * delt);
	  
      K = P1*H/( H*P1*H + U*U);

      z = log_stock_prices[i1+1];

      x = x1 + K * (z - (log_stock_prices[i1] + (muS-0.5*x1)*delt)); //KF: measurement updates 2.9
      u[i1] = z - (log_stock_prices[i1] + (muS-0.5*x1)*delt); 
      v[i1] = H*P1*H + U*U; 
      estimates[i1+1] = log_stock_prices[i1] + (muS-0.5*x1)*delt; 
      P=(1.0-K*H)*P1; 
	}
}

/*The codes on p.105 (inf 2ed)
 *The concepts on p.76 (inf 2ed): steps of unscented kalman filters
 *1. initial choice for the state vector \hat x_0  and
 *
 */
void estimate_unscented_kalman_parameters_1_dim(double *log_stock_prices,
                                                double muS,
                                                int n_stock_prices,
                                                double omega,
                                                double theta, double xi,
                                                double rho,
                                                double p,
                                                double *u,
                                                double *v,
                                                double *estimates)
{
  int     i1,i2, i3, t1;
  int     ret;
  int     na=3; //1-b n_a
  double  x, xa[3];  // state x, state space Augmentation 
  double  X[7], Xa[3][7]; //sigma point, Unscented Transformation
  double  Wm[7], Wc[7], Z[7]; //weighted of mean, weighted of  covariance , observation
  double  x1; // step 2 \mu
  double  prod, prod1; //prod step 2 \sigma
  double  P, P1;  // value step 3-b P_k->measurement covariance; P1-> P^-_k  
  double **Pa, **proda; // matries step 3-b Pa->measurement covariance; P1-> P^-_k    
  double  z, U, Pzz, K; // step 3-b measurement \hat z_k, Pzz->P_z_kz_k, K->Kalman gain 
  double  delt=1.0/252.0; //\Delta t (time)
  double  a=0.001 , b=0.0, k=0.0, lambda; //\alpha, \beta, \kama and \lambda
  double eps=0.00001; //error 
  //step (1-a) 
  lambda = a*a*(na +k)-na;
  proda= new double * [na];
  Pa =   new double * [na];
  for (i1=0;i1<na;i1++)
    {
      Pa[i1]= new double [na];
      proda[i1]= new double [na];
    }
  xa[1]=xa[2]=0.0;
  x= 0.04;
  u[0]=u[n_stock_prices-1]=0.0;
  v[0]=v[n_stock_prices-1]=1.0;
  estimates[0]=estimates[1]=log_stock_prices[0]+eps;
  xa[0]=x;
  Pa[0][0]= Pa[1][1]= Pa[2][2] = 1.0;
  Pa[1][0]= Pa[0][1]= Pa[1][2]=Pa[2][1]= Pa[0][2]=Pa[2][0]=0;
  for (i1=0;i1<na;i1++)
    {
      for (i2=0;i2<na;i2++)
        {
          proda[i1][i2]=0.0;
        }
    }

  //(1-b)for (step 2)
  Wm[0]=lambda/(na+lambda);
  Wc[0]=lambda/(na+lambda) + (1-a*a+b);
  for (i3=1;i3<(2*na+1);i3++)
    {
      Wm[i3]=Wc[i3]=1/(2*(na+lambda));
    }
  for (t1=1;t1<n_stock_prices-1;t1++)
    {
      for (i1=0;i1<na;i1++)
        {
          Xa[i1][0]= xa[i1];
        }
      for (i1=0;i1<na;i1++)
        {
          for (i2=0;i2<na;i2++)
            {
              if (i1==i2)
                {
                  if (Pa[i1][i2] < 1.0e-10)
                    Pa[i1][i2]= 1.0e-10;
                } else {
                if (Pa[i1][i2] < 1.0e-10)
                  Pa[i1][i2]= 0.0;
              }
            }
        }
      ret = sqrt_matrix(Pa, proda, na);
      // step 1-d, Unscented Transformation, prepares for the time update (predict)
      for (i3=1;i3<(1+na);i3++) //i^{th} columns of the square-root matrix 
        {
          for (i1=0;i1<na;i1++)
            {
              Xa[i1][i3]= xa[i1] + sqrt(na+lambda) * proda[i1][i3-1];
            }
        }
      for (i3=(1+na);i3<(2*na+1);i3++) //i-n_a^{th} columns of the square-root matrix 
	  {
          for (i1=0;i1<na;i1++)
            {
              Xa[i1][i3]= xa[i1] - sqrt(na+lambda) * proda[i1][i3-na-1];
            } 
	  }
	  
	  // step 2 the time update equations (predict)
      for (i3=0;i3<(2*na+1);i3++) //sigma point X_{k \mid k-1}(i)
        {
          if (Xa[0][i3]<0) Xa[0][i3]=0.0001;
         /* //p.102 formula 2.15
          X[i3]= Xa[0][i3] + (omega-muS*rho*xi - (theta-0.5*rho*xi) *Xa[0][i3])*delt +
            rho*xi* (log_stock_prices[t1]-log_stock_prices[t1-1]) +
            xi*sqrt((1-rho*rho)*delt*Xa[0][i3])*Xa[1][i3];
         */
         // p.134 formula 2.27  p=1 heaston model; p=0.5 GARCH model; p=1.5 3/2 model
          X[i3]= Xa[0][i3] + (omega-muS*rho*xi*pow(Xa[0][i3], p - 0.5) - (theta-0.5*rho*xi*pow(Xa[0][i3], p - 0.5) ) *Xa[0][i3])*delt +
            rho*xi*pow(Xa[0][i3], p - 0.5)*(log_stock_prices[t1]-log_stock_prices[t1-1]) +
            xi*sqrt((1-rho*rho)*delt)*pow(Xa[0][i3], p)*Xa[1][i3];
        }
      x1 = 0;
      for (i3=0;i3<(2*na+1);i3++) // sigma point mean
        {
          x1 += Wm[i3]*X[i3];
        }
      P1=0.0;
      for (i3=0;i3<(2*na+1);i3++) // sigma point covariance
        {
          P1 += Wc[i3]*(X[i3]-x1)*(X[i3]-x1);
        }
        
      //step 3  
      z=0;
      for (i3=0;i3<(2*na+1);i3++) // 3-a innovation
        {
          if (X[i3]<0) X[i3]=0.00001;
          Z[i3] = log_stock_prices[t1] + (muS-0.5*X[i3])*delt +
            sqrt(X[i3]*delt)*Xa[2][i3];
          z += Wm[i3]*Z[i3];
        }
        
      Pzz=0; //(3-b) Measurement Update
      for (i3=0;i3<(2*na+1);i3++) //P_{z_k z_k}
        {
          Pzz +=  Wc[i3]*(Z[i3]-z)*(Z[i3]-z);
        }
      prod=0.0;
      for (i3=0;i3<(2*na+1);i3++) //p_{x_k x_k}
        {
          prod += Wc[i3]*(X[i3]-x1)* (Z[i3]-z);
        }
      K= prod/Pzz;    //Kalman gain
      u[t1] = log_stock_prices[t1+1] - z;
      v[t1] = Pzz;
      estimates[t1+1] = z;
      x = x1 + K*(log_stock_prices[t1+1] - z); //measurement mean
      P = P1 - K*K * Pzz;   //measrement covariance
      xa[0]=x;   //update matries
      Pa[0][0] = P; //update matriex
      if (x<0) x=0.0001;
      Pa[1][0]= Pa[0][1]= Pa[1][2]=Pa[2][1]= Pa[0][2]=Pa[2][0]=0;
    }
  for (i1=0;i1<na;i1++)
    {
      delete [] Pa[i1];
      delete [] proda[i1];
    }
  delete [] Pa;
  delete [] proda;
}


// the routine sqrt_matrix() can be constructed via the Cholesly decomposition
// also available as choldc() in the Numerical Recipes in C
// log_stock_prices  are the log of stock prices
// muS is the real-world stock drift
// n_stock_prices is the number of the above stock prices
// (omega, theta, xi, rho) are the Heston parameters
// ll is the value of (negative log) Likelihood function
// estimates[] are the estimated observations from the filter
// The function ran2() is from Numerical Recipes in C
// and generates uniform random variables
// The function Normal_inverse() can be found from many sources
// and is the inverse of the Normal CDF
// Normal_inverse(ran2(.)) generates a set of Normal random variables
void estimate_particle_extended_kalman_parameters_1_dim(
                                                        double *log_stock_prices,
                                                        double muS,
                                                        int n_stock_prices,
                                                        double omega,
                                                        double theta,
                                                        double xi,
                                                        double rho,
                                                        double *ll,
                                                        double *estimates)
{
  int     i1, i2, i3;
  double  H, A, x0, P0, z;
  int     M=1000;
  double  x[1000], xx[1000], x1[1000], x2[1000];
  double  P[1000], P1[1000], U[1000], K[1000], W[1000];
  double  w[1000],  u[1000], c[1000];
  double  q, pz, px, s, m, l;
  double  delt=1.0/252.0, x1_sum;
  //long  idum=-1;
  int     idum = -1;
  A = 1.0-(theta-0.5*rho*xi)*delt;
  H = -0.5*delt;
  x0 = 0.04;
  P0 = 0.000001;
  for (i2=0; i2<M; i2++)
    {
      x[i2] = x0 + sqrt(P0)* cumulative_normal(get_ran2(idum)); //(my code)NormalCDFInverse(get_ran2(idum)); //(original code)Normal_inverse(ran2(&idum));
      P[i2] = P0;
    }
  *ll=0.0;
  for (i1=1;i1<n_stock_prices-1;i1++)
    {
      l = 0.0;
      x1_sum=0.0;
      for (i2=0; i2<M; i2++)
        {
          //2. EKF for the proposal distribution 
          if (x[i2]<0) x[i2]=0.00001;
          x1[i2] = x[i2] + ( omega-rho*xi*muS - (theta-
				   0.5*rho*xi) * x[i2]) * delt + rho*xi*
				   (log_stock_prices[i1]-log_stock_prices[i1-1]);
          W[i2]  = xi*sqrt((1-rho*rho) * x[i2] * delt);
          P1[i2] = W[i2]*W[i2] + A*P[i2]*A;
          if (x1[i2]<0) x1[i2]=0.00001;
          U[i2] = sqrt(x1[i2]*delt);
          K[i2] = P1[i2]*H/( H*P1[i2]*H + U[i2]*U[i2]);
          z = log_stock_prices[i1+1];
          x2[i2] = x1[i2] + K[i2] * (z - (log_stock_prices[i1]
                                          + (muS-0.5*x1[i2])*delt));
          x1_sum+= x1[i2];
          P[i2]=(1.0-K[i2]*H)*P1[i2];
          //3. sample 
          xx[i2] = x2[i2]+sqrt(P[i2])* cumulative_normal(get_ran2(idum)); //(my code)NormalCDFInverse(get_ran2(idum));  //(original code)Normal_inverse(ran2(&idum));
          if (xx[i2]<0) xx[i2]=0.00001;
          //4. calculate weights 
          m = x2[i2];
          s = sqrt(P[i2]);
          q = 0.39894228/s * exp( - 0.5* (xx[i2] - m)*
                                  (xx[i2] - m)/(s*s) );
          m = log_stock_prices[i1] + (muS-0.5*xx[i2])*delt;
          s = sqrt(xx[i2]*delt);
          pz = 0.39894228/s * exp( - 0.5* (z - m)*(z - m)/(s*s) );
          m = x[i2] + ( omega-rho*xi*muS - (theta-0.5*
                                            rho*xi) * x[i2]) * delt + rho*xi*
            (log_stock_prices[i1]-log_stock_prices[i1-1]);
          s = xi*sqrt((1-rho*rho) * x[i2] * delt);
          px = 0.39894228/s * exp( - 0.5* (xx[i2] - m)*
                                   (xx[i2] - m)/(s*s) );
          w[i2] = pz * px / MAX(q, 1.0e-10);
          l += w[i2]; }
      *ll += log(l);
      estimates[i1+1]= log_stock_prices[i1] +
        (muS-0.5*x1_sum/M)*delt;
      //5. normalize weights 
      for (i2=0; i2<M; i2++)
        w[i2] /= l;
      //6. resample and reset weights 
      c[0]=0;
      for (i2=1; i2<M; i2++)
        c[i2] = c[i2-1] + w[i2];
      i2=0;
      u[0] = 1.0/M * get_ran2(idum); //ran2(&idum);
      for (i3=0; i3<M; i3++)
        {
          u[i3] = u[0] + 1.0/M *i3;
          while (u[i3] > c[i2])
            i2++;
          x[i3] = xx[i2];
          w[i3] = 1.0/M;
        }
    }
  *ll *= -1.0; }
// *ll is the value of (negative log) Likelihood function
// we can minimize it to obtain the optimal parameter-set


void estimate_particle_unscented_kalman_parameters_1_dim(
                                                         double *log_stock_prices,
                                                         double muS,
                                                         int n_stock_prices,
                                                         double omega,
                                                         double theta,
                                                         double xi,
                                                         double rho,
                                                         double *ll,
                                                         double *estimates)
{
  int      i1, i2, i3, i4;
  int      na=3;
  double   x0, P0;
  double   Wm[7], Wc[7];
  int      M=1000;
  double   x[1000], xx[1000], x1[1000], x2[1000],
           zz[1000], Z[1000][7];
  double   X[1000][7], Xa[1000][3][7];
  double   xa[1000][3], prod[1000];
  double   P[1000], P1[1000], U[1000], K[1000],
           W[1000], Pzz[1000];
  double   w[1000], u[1000], c[1000]; double ***Pa, ***proda;
  double   q,pz,px,s,m, l, z;
  double   delt=1.0/252.0;
  //long idum=-1;
  int idum=-1;
  int ret;
  double a=0.001 , b=0.0, k=0.0, lambda;
  proda= new double ** [M];
  Pa =   new double ** [M];
  for (i2=0;i2<M;i2++)
    {
      Pa[i2]= new double * [na];
      proda[i2]= new double * [na];
      for (i1=0;i1<na;i1++)
        {
          Pa[i2][i1]= new double [na];
          proda[i2][i1]= new double [na];
        }
    }
  for (i2=0;i2<M;i2++)
    {
      for (i1=0;i1<na;i1++)
        {
          for (i3=0;i3<na;i3++)
            {
              proda[i2][i1][i3]=0.0;
            } }
    }
  lambda = a*a*(na +k)-na;
  Wm[0]=lambda/(na+lambda);
  Wc[0]=lambda/(na+lambda) + (1-a*a+b);
  for (i3=1;i3<(2*na+1);i3++)
    {
      Wm[i3]=Wc[i3]=1/(2*(na+lambda));
    }
  x0 = 0.04;
  P0 = 0.000001;
  for (i2=0; i2<M; i2++)
    {
      x[i2] = x0 + sqrt(P0)* cumulative_normal(get_ran2(idum));// Normal_inverse(ran2(&idum));
      P[i2] = P0;
      xa[i2][0]=x[i2];
      xa[i2][1]=xa[i2][2]=0.0;
      Pa[i2][0][0]= P[i2];
      Pa[i2][1][1]= Pa[i2][2][2] = 1.0;
      Pa[i2][1][0]= Pa[i2][0][1]= Pa[i2][1][2] =
        Pa[i2][2][1] = Pa[i2][0][2] = Pa[i2][2][0] = 0.0;
    }
  *ll=0.0;
  for (i1=1;i1<n_stock_prices-1;i1++)
    {
      l = 0.0;
      estimates[i1+1]=0.0;
      for (i2=0; i2<M; i2++)
        {
          //2. UKF for the proposal distribution 
          for (i3=0;i3<na;i3++)
            {
              Xa[i2][i3][0]= xa[i2][i3];
            }
          for (i3=0;i3<na;i3++)
            {
              for (i4=0;i4<na;i4++)
                {
                  if (i3==i4)
                    {
                      if (Pa[i2][i3][i4] < 1.0e-10)
                        Pa[i2][i3][i4]= 1.0e-10;
                    } else {
                    if (Pa[i2][i3][i4] < 1.0e-10)
                      Pa[i2][i3][i4] = 0.0;
                  } }
            }
          ret = sqrt_matrix(Pa[i2],proda[i2],na);
          for (i3=1;i3<(1+na);i3++)
            {
              for (i4=0;i4<na;i4++)
                {
                  Xa[i2][i4][i3]= xa[i2][i4] + sqrt(na+lambda) *
                    proda[i2][i4][i3-1];
                } }
          for (i3=(1+na);i3<(2*na+1);i3++)
            {
              for (i4=0;i4<na;i4++)
                {
                  Xa[i2][i4][i3]= xa[i2][i4] - sqrt(na+lambda) *
                    proda[i2][i4][i3-na-1];
                } }
          for (i3=0;i3<(2*na+1);i3++)
            {
              if (Xa[i2][0][i3]<0) Xa[i2][0][i3]=0.0001;
              X[i2][i3]= Xa[i2][0][i3] + (omega-muS*rho*xi   -
                                          (theta-0.5*rho*xi) *Xa[i2][0][i3])*delt +
                rho*xi* (log_stock_prices[i1]-
                         log_stock_prices[i1-1]) +
                xi*sqrt((1-rho*rho)*delt*Xa[i2][0][i3])*
                Xa[i2][1][i3];
            }
          x1[i2] = 0;
          for (i3=0;i3<(2*na+1);i3++)
            {
              x1[i2] += Wm[i3]*X[i2][i3];
            }
          P1[i2]=0.0;
          for (i3=0;i3<(2*na+1);i3++)
            {
              P1[i2] += Wc[i3]*(X[i2][i3]-x1[i2])*(X[i2][i3]-x1[i2]);
            }
          zz[i2]=0;
          for (i3=0;i3<(2*na+1);i3++)
            {
              if (X[i2][i3]<0) X[i2][i3]=0.00001;
              Z[i2][i3] = log_stock_prices[i1] +
                (muS-0.5*X[i2][i3])*delt + sqrt(X[i2][i3]*delt)*Xa[i2][2][i3];
              zz[i2] += Wm[i3]*Z[i2][i3];
            }
          Pzz[i2]=0;
          for (i3=0;i3<(2*na+1);i3++)
            {
              Pzz[i2] +=  Wc[i3]*(Z[i2][i3]-zz[i2])*(Z[i2][i3]-zz[i2]);
            }
          prod[i2]=0.0;
          for (i3=0;i3<(2*na+1);i3++)
            {
              prod[i2] += Wc[i3]*(X[i2][i3]-x1[i2])* (Z[i2][i3]-zz[i2]);
            }
          K[i2]= prod[i2]/Pzz[i2];
          z = log_stock_prices[i1+1];
          estimates[i1+1] += zz[i2]/M;
          x2[i2] = x1[i2] + K[i2]*(z - zz[i2]);
          P[i2] = P1[i2] - K[i2]*K[i2] * Pzz[i2];
          xa[i2][0]=x2[i2];
          Pa[i2][0][0] = P[i2];
          if (x2[i2]<0) x2[i2]=0.0001;
          Pa[i2][1][0]= Pa[i2][0][1]= Pa[i2][1][2]
            =Pa[i2][2][1]= Pa[i2][0][2]=Pa[i2][2][0]=0;
          //3. sample 
          xx[i2] = x2[i2] + sqrt(P[i2])*cumulative_normal(get_ran2(idum)); //Normal_inverse(ran2(&idum));
          if (xx[i2]<0) xx[i2]=0.00001;
          //4. calculate weights 
          m = x2[i2];
          s = sqrt(P[i2]);
          q = 0.39894228/s * exp( - 0.5* (xx[i2] - m)*
                                  (xx[i2] - m)/(s*s) );
          m= log_stock_prices[i1] + (muS-0.5*xx[i2])*delt;
          s= sqrt(xx[i2]*delt);
          pz= 0.39894228/s * exp( - 0.5* (z - m)*
                                  (z - m)/(s*s) );
          m= x[i2] + ( omega-rho*xi*muS -
                       (theta-0.5*rho*xi) * x[i2]) * delt +
            rho*xi* (log_stock_prices[i1]-
                     log_stock_prices[i1-1]);
          s= xi*sqrt((1-rho*rho) * x[i2] * delt);
          px= 0.39894228/s * exp( - 0.5* (xx[i2] - m)*
                                  (xx[i2] - m)/(s*s) );
          w[i2]= MAX(pz, 1.0e-10) *
            MAX(px, 1.0e-10) / MAX(q, 1.0e-10);
          l += w[i2];
        }
      *ll += log(l);
      //5. normalize weights 
      for (i2=0; i2<M; i2++)
        w[i2] /= l;
      //6. resample and reset weights 
      c[0]=0;
      for (i2=1; i2<M; i2++)
        c[i2] = c[i2-1] + w[i2];
      i2=0;
      u[0] = 1.0/M * get_ran2(idum); //ran2(&idum);
      for (i3=0; i3<M; i3++)
        {
          u[i3] = u[0] + 1.0/M *i3;
          while (u[i3] > c[i2])
            i2++;
          x[i3]= xx[i2];
          w[i3]=1.0/M;
        }
    }
  *ll *= -1.0;
  for (i2=0;i2<M;i2++)
    {
      for (i1=0;i1<na;i1++)
        {
          delete [] Pa[i2][i1];
          delete [] proda[i2][i1];
        }
    }
  for (i2=0;i2<M;i2++)
    {
      delete [] Pa[i2];
      delete [] proda[i2];
    }
  delete [] Pa;
  delete [] proda;
}


/*
 p.179
 log_stock_prices are the log of stock prices
 muS is the real-world stock drift
 n_stock_prices is the number of the above stock prices
 (omega, theta, xi, rho, lambda, j) are the Bates parameters
 ll is the value of (negative log) Likelihood function
 estimates[] are the estimated observations from the filter
 The function ran2() is from Numerical Recipes in C
 and generates uniform random variables
 The function Normal_inverse() can be found from many sources
 and is the inverse of the Normal CDF
 Normal_inverse(ran2(.)) generates a set of Normal random variables
*/
void estimate_particle_jump_diffusion_parameters_1_dim(
													   double *log_stock_prices,
													   double muS, int n_stock_prices,
													   double omega,
													   double theta,
													   double xi,
													   double rho,
													   double lambda,
													   double j,
													   double *ll,
													   double *estimates)
 
{
 int i1, i2, i3;
 double H, A, x0, P0, z;
 int M=1000;
 double x[1000], xx[1000], x1[1000], x2[1000];
 double mu[1000], mm[1000], m1[1000], m2[1000];
 double P[1000], P1[1000], U[1000], K[1000], W[1000];
 double w[1000], u[1000], c[1000];
 double q, pz, px, s, m, l;
 double delt=1.0/252.0, x1_sum, m1_sum;
 //long idum=-1;
 int idum=-1;
 int i1_prev=0;
 double u_t=0.0;
 int *jump;
 jump= new int [n_stock_prices];
 for (i1=0; i1<n_stock_prices; i1++)
	jump[i1]=0;
 A = 1.0-(theta-0.5*rho*xi)*delt;
 H = -0.5*delt;
 x0 = 0.04;
 P0 = 0.000001;
 for (i2=0; i2<M; i2++)
 {
	x[i2] = x0 + sqrt(P0)* cumulative_normal(get_ran2(idum));//Normal_inverse(ran2(&idum));
	mu[i2]=0;
	P[i2] = P0;
 }
 
 *ll=0.0;
 for (i1=1;i1<n_stock_prices-1;i1++)
 {
	l = 0.0;
	x1_sum=0.0;
	m1_sum=0.0;
		for (i2=0; i2<M; i2++)
		{
 /* EKF for the proposal distribution */
			if (x[i2]<0) x[i2]=0.00001;
				x1[i2] = x[i2] + ( omega-rho*xi*(muS+lambda*j) -
				(theta-0.5*rho*xi) * x[i2]) * delt +
				rho*xi* (log_stock_prices[i1]-log_stock_prices[i1-1])
				- rho*xi*mu[i2];
			m1[i2]=mu[i2];
			W[i2] = xi*sqrt((1-rho*rho) * x[i2] * delt);
			P1[i2] = W[i2]*W[i2] + A*P[i2]*A;
			if (x1[i2]<0) x1[i2]=0.00001;
				U[i2] = sqrt(x1[i2]*delt);
				K[i2] = P1[i2]*H/( H*P1[i2]*H + U[i2]*U[i2]);
				z = log_stock_prices[i1+1];
				x2[i2] = x1[i2] + K[i2] *
				(z - (log_stock_prices[i1] + (muS+lambda*j-0.5*x1[i2])*delt
				+ m1[i2]));
			m2[i2]= m1[i2];
			x1_sum+= x1[i2];
			m1_sum+= m1[i2];
			P[i2]=(1.0-K[i2]*H)*P1[i2];
 /* sample */
			xx[i2] = x2[i2] + sqrt(P[i2])* cumulative_normal(get_ran2(idum)); //Normal_inverse(ran2(&idum));
			if (xx[i2]<0) xx[i2]=0.00001;
				//if (ran2(&idum) < exp(-lambda*delt))
                if (get_ran2(idum) < exp(-lambda*delt))
					mm[i2]=0.0;
				else
					mm[i2]=log(1.0-j);
 /* calculate weights */
				m = x2[i2];
				s = sqrt(P[i2]);
				q = 0.39894228/s * exp( - 0.5* (xx[i2] - m)*(xx[i2] - m)/(s*s) );
				m= log_stock_prices[i1] + (muS+lambda*j-0.5*xx[i2])*delt + mm[i2];
				s= sqrt(xx[i2]*delt);
				pz= 0.39894228/s * exp( - 0.5* (z - m)*(z - m)/(s*s) );
				m= x[i2] + ( omega-rho*xi*(muS+lambda*j) -
					(theta-0.5*rho*xi) * x[i2]) * delt +
					rho*xi* (log_stock_prices[i1]-log_stock_prices[i1-1])
					-rho*xi*mu[i2];
				s= xi*sqrt((1-rho*rho) * x[i2] * delt);
				px= 0.39894228/s * exp( - 0.5* (xx[i2] - m)*(xx[i2] - m)/(s*s) );
				w[i2]= pz * px / MAX(q, 1.0e-10);
				l += w[i2];
		}
		
	*ll += log(l);
	estimates[i1+1]= log_stock_prices[i1] +
	(muS+lambda*j-0.5*x1_sum/M)*delt+m1_sum/M;
 /* normalize weights */
	for (i2=0; i2<M; i2++)
		w[i2] /= l;
 /* resample and reset weights */
		c[0]=0;
		for (i2=1; i2<M; i2++)
			c[i2] = c[i2-1] + w[i2];
			i2=0;
			u[0] = 1.0/M * get_ran2(idum);//ran2(&idum);
			for (i3=0; i3<M; i3++)
			{
				u[i3] = u[0] + 1.0/M *i3;
				while (u[i3] > c[i2])
					i2++;
				x[i3]= xx[i2];
				mu[i3]=mm[i2];
				w[i3]=1.0/M;
			}
 }
 
 *ll *= -1.0;
 delete [] jump;
}
// *ll corresponds to the (negative log) Likelihood
// which we will need to minimize to obtain optimal parameters


/*
 p. 189
 log_stock_prices are the log of stock prices
 muS is the real-world stock drift
 n_stock_prices is the number of the above stock prices
 (kappa,eta,lambda,sigma,theta,nu) are the VGSA parameters
 ll is the value of (negative log) Likelihood function
 estimates[] are the estimated observations from the filter
 errors are the observation errors
 The function ran2() is from Numerical Recipes in C
 and generates uniform random variables
 The function Normal_inverse() can be found from many sources
 and is the inverse of the Normal CDF
 Normal_inverse(ran2(.)) generates a set of Normal random
variables
 The Bessel and Gamma functions bessik() and gammln()
 are also available in Numerical Recipes in C
*/
void estimate_particle_extended_VGSA_parameters_bessel(
													   double *log_stock_prices,
													   double muS,
													   int n_stock_prices,
													   double kappa,
													   double eta,
													   double lambda,
													   double sigma,
													   double theta,
													   double nu,
													   double *ll,
													   double *estimates,
													   double *errors)
 
{
 int i1, i2, i3;
 double y0, z, omega;
 int
 M=1000;
 double x[1000], xx[1000], X;
 double w[1000], u[1000], c[1000];
 double pz, px, q, s, m, l, x1_sum;
 //long idum=-1;
 int idum=-1;
 double delt=1.0/252.0;
 double eps=1.0e-30;
 double Ka,Ia,Kp,Ip, Kx,Knu;
 double H, A, x0, P0;
 double P[1000], P1[1000], U[1000], K[1000], W[1000];
 double x1[1000], x2[1000];
 /* initialize */
 omega=log(1.0-theta*nu- sigma*sigma*nu/2.0)/nu;
 x0 = 1.0;
 P0 = 0.000001;
 for (i2=0; i2<M; i2++)
 {
	x[i2] = x0 + sqrt(P0)* cumulative_normal(get_ran2(idum)); //Normal_Inverse(ran2(&idum));
	P[i2] = P0;
 }
 A = 1.0-kappa*delt;
 H = theta*delt;
 /* time loop */
 *ll=0.0;
 for (i1=1;i1<n_stock_prices-1;i1++)
 {
	z = log_stock_prices[i1+1]-log_stock_prices[i1];
	X= z - muS*delt - delt/nu*log(1.0-theta*nu- sigma*sigma*nu/2.0);
	l = 0.0;
	x1_sum=0.0;
	for (i2=0; i2<M; i2++)
	{
 /* EKF for the proposal distribution */
		x1[i2] = x[i2] + kappa*(eta - x[i2])*delt;
		W[i2] = lambda*sqrt(x[i2] * delt);
		P1[i2] = W[i2]*W[i2] + A*P[i2]*A;
		x1[i2]=MAX(x1[i2],eps);
		U[i2] = sqrt(theta*theta*nu+sigma*sigma)*sqrt(x1[i2]*delt);
		K[i2] = P1[i2]*H/( H*P1[i2]*H + U[i2]*U[i2]);
		x2[i2] = x1[i2] + K[i2] * (z - (muS+omega+theta*x1[i2])*delt);
		x1_sum+= x1[i2];
		P[i2]=(1.0-K[i2]*H)*P1[i2];
 /* sample */
		xx[i2] = x2[i2] + sqrt(P[i2])* cumulative_normal(get_ran2(idum));  //Normal_Inverse(ran2(&idum));
		xx[i2]=MAX(xx[i2],eps);
 /* calculate weights */
		m = x2[i2];
		s = sqrt(P[i2]);
		q = 0.39894228/s * exp( - 0.5* (xx[i2] - m)*(xx[i2] - m)/(s*s) );
		m = x[i2] + kappa*(eta - x[i2])*delt;
		s = lambda*sqrt(x[i2] * delt);
		px = 0.39894228/s * exp( - 0.5* (xx[i2] - m)*(xx[i2] - m)/(s*s) );
		Kx = MAX(eps, 1.0/(sigma*sigma)*sqrt(X*X*(2*sigma*sigma/nu+theta*theta)));
		Knu = MAX(eps, (xx[i2]*delt/nu-0.5));
		//bessik(Kx , Knu , &Ia, &Ka, &Kp, &Ip);
		get_bessik(Kx , Knu , Ia, Ka, Kp, Ip);
		pz=2.0*exp(theta*X/(sigma*sigma)) /
		//(pow(nu,xx[i2]*delt/nu)*sigma*exp(gammln(xx[i2]*delt/nu)))
		(pow(nu,xx[i2]*delt/nu)*sigma*exp(get_gammln(xx[i2]*delt/nu)))
		*0.39894228*pow(X*X/(2*sigma*sigma/nu+theta*theta), 0.5*xx[i2]*delt/nu-0.25)* Ka;
		w[i2]= pz * px / MAX(q, eps);
		l += w[i2];
	}
	
	*ll += log(l);
	
 /* estimates[i1+1] for z[i1] => error term */
	estimates[i1+1]= log_stock_prices[i1+1]-
	(log_stock_prices[i1] + (muS+omega+theta*x1_sum/M)*delt);
	errors[i1] = (theta*theta*nu + sigma*sigma)* x1_sum/M*delt;
 
 /* normalize weights */
	for (i2=0; i2<M; i2++)
		w[i2] /= l;
 
 /* resample and reset weights */
	c[0]=0;
	
	for (i2=1; i2<M; i2++)
		c[i2] = c[i2-1] + w[i2];
		i2=0;
		u[0] = 1.0/M * get_ran2(idum);//ran2(&idum);
		
		for (i3=0; i3<M; i3++)
		{
			u[i3] = u[0] + 1.0/M *i3;
			while (u[i3] > c[i2])
				i2++;
			
			x[i3]= xx[i2];
			w[i3]=1.0/M;
		}
 }
 
 *ll *= -1.0;
}
// *ll represents the (negative log) Likelihood
