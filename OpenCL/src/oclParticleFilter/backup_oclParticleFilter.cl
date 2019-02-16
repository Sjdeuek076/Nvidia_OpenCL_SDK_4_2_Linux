
__kernel void estimate_particle_unscented_kalman_parameters_1_dim(__global float *log_stock_prices,
                                                                  __global float *ll,
                                                                  __global float *estimates,
                                                                  float omega,
                                                                  float theta,
                                                                  float xi,
                                                                  float rho,
                                                                  float muS,
                                                                  int n_stock_prices)

{
  //int     i1 = get_global_id(0); 
  int 	  i2, i3, i4;
  int     na=3;
  float   x0, P0;
  float   Wm[7], Wc[7];
  int     M=1000;
  float   x[1000], xx[1000], x1[1000], x2[1000], zz[1000], Z[1000][7];
  float   X[1000][7], Xa[1000][3][7];
  float   xa[1000][3], prod[1000];
  float   P[1000], P1[1000], U[1000], K[1000], W[1000], Pzz[1000];
  float   w[1000], u[1000], c[1000]; float ***Pa, ***proda;
  float   q,pz,px,s,m, l, z;
  float   delt=1.0/252.0;
  //long idum=-1;
  int idum=-1;
  int ret;
  float a=0.001 , b=0.0, k=0.0, lambda;
  proda= new float ** [M];
  Pa =   new float ** [M];
  for (i2=0;i2<M;i2++)
    {
      Pa[i2]= new float * [na];
      proda[i2]= new float * [na];
      for (i1=0;i1<na;i1++)
        {
          Pa[i2][i1]= new float [na];
          proda[i2][i1]= new float [na];
        }
    }
  for (i2=0;i2<M;i2++)
    {
      for (i1=0;i1<na;i1++)
        {
          for (i3=0;i3<na;i3++)
            {
              proda[i2][i1][i3]=0.0;
            } 
        }
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
      Pa[i2][1][0]= Pa[i2][0][1]= Pa[i2][1][2] 
       =Pa[i2][2][1] = Pa[i2][0][2] = Pa[i2][2][0] = 0.0;
    }
  *ll=0.0;
  
  // ukf begin
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
                    }else 
                    {
                    if (Pa[i2][i3][i4] < 1.0e-10)
                      Pa[i2][i3][i4] = 0.0;
                    }
                }
            }
          ret = sqrt_matrix(Pa[i2],proda[i2],na);
          for (i3=1;i3<(1+na);i3++)
            {
              for (i4=0;i4<na;i4++)
                {
                  Xa[i2][i4][i3]= xa[i2][i4] + sqrt(na+lambda) *
                    proda[i2][i4][i3-1];
                } 
            }
          for (i3=(1+na);i3<(2*na+1);i3++)
            {
              for (i4=0;i4<na;i4++)
                {
                  Xa[i2][i4][i3]= xa[i2][i4] - sqrt(na+lambda) *
                    proda[i2][i4][i3-na-1];
                } 
            }
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
        } //end M loop 
        
        
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
        
    }// end ukf //end n_stock_prices loop
    
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







/*__kernel void estimate_unscented_kalman_parameters_1_dim(__global float *log_stock_prices,
		                                                 __global float *u,
		                                                 __global float *v,
		                                                 __global float *estimates,                                                                                  
		                                                 float omega,
		                                                 float theta, 
		                                                 float xi,
		                                                 float rho,
		                                                 float muS, 
		                                                 float p,
		                                                 int n_stock_prices)
{
  int     i1,i2, i3, t1;
  int     ret;
  int     na=3; //1-b n_a
  float  x, xa[3];  // state x, state space Augmentation 
  float  X[7], Xa[3][7]; //sigma point, Unscented Transformation
  float  Wm[7], Wc[7], Z[7]; //weighted of mean, weighted of  covariance , observation
  float  x1; // step 2 \mu
  float  prod, prod1; //prod step 2 \sigma
  float  P, P1;  // value step 3-b P_k->measurement covariance; P1-> P^-_k  
  float  **Pa, **proda; // matries step 3-b Pa->measurement covariance; P1-> P^-_k    
  float  z, U, Pzz, K; // step 3-b measurement \hat z_k, Pzz->P_z_kz_k, K->Kalman gain 
  float  delt=1.0/252.0; //\Delta t (time)
  float  a=0.001 , b=0.0, k=0.0, lambda; //\alpha, \beta, \kama and \lambda
  float eps=0.00001; //error 
  //step (1-a) 
  lambda = a*a*(na +k)-na;
  proda= new float * [na];
  Pa =   new float * [na];
  for (i1=0;i1<na;i1++)
    {
      Pa[i1]= new float [na];
      proda[i1]= new float [na];
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
         // //p.102 formula 2.15
         // X[i3]= Xa[0][i3] + (omega-muS*rho*xi - (theta-0.5*rho*xi) *Xa[0][i3])*delt +
         //   rho*xi* (log_stock_prices[t1]-log_stock_prices[t1-1]) +
         //   xi*sqrt((1-rho*rho)*delt*Xa[0][i3])*Xa[1][i3];
         
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
*/



/*__kernel void estimate_ekf_parm_1_dim_heston(__global float *log_stock_prices,                                             
                                             __global float *u,
                                             __global float *v,                                             
                                             __global float *estimates,                                                                                                                                   
                                             float omega,
                                             float theta,
                                             float xi,
                                             float rho,
                                             float muS,
                                             int n_stock_prices)                                                                                          
{
  //int i1;
  //uint num_rows = get_global_size(0); //global size = n_stock_prices
  int i1 = get_global_id(0);
  float x, x1, W, H, A;
  float P, P1, z, U, K;
  const float delt=1.0/252.0;
  const float eps=0.00001;
  x=0.04;
  P=0.01;
  //u[0]=u[n_stock_prices-1]=0.0;
  //v[0]=v[n_stock_prices-1]=1.0;
  //estimates[0]=estimates[1]=log_stock_prices[0]+eps;
  //for (i1=1;i1<n_stock_prices-1;i1++)
   if (i1 < n_stock_prices) 
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
}*/





