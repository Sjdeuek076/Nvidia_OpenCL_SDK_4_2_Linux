#include <cmath>
#include "nr.h"
using namespace std;
/* NR p.413 Powell's conjugate direction method is an algorithm for finding a local minimum of a function
 * Powell first discovered a direction set method that does produce N mutually conjugate directions.(10.5.1)
 * repeat the followings sequence of steps until your function stops decreasing
 * 1. Save your starting position as P_0
 * 2. For i=1,...N, move P_{i-1} to the minimum along direction u_i and call this point P_i
 * 3. For i=1,...,N-1, set u_i <- u_{i+1}
 * 4. set u_N <- P_N - P_0
 * 5. Move P_N to the minimum along direction u_N and call this pint P_0
 */
void NR::powell(Vec_IO_DP &p, Mat_IO_DP &xi, const DP ftol, int &iter,
	DP &fret, DP func(Vec_I_DP &))
// Vec_IO_DP: initial starting point p[1..n]; On output (Vec_IO_DP is set to the best point found)
// Mat_IO_DP: initial matrix xi[1..n][1..n], whose columns contain the initial set of directions  (Mat_IO_DP is the then-current direction set)
// ftol: the fractional tolerance in the function value such that failure to decrease by more than this amount on one iteration signals doneness.
// iter: is the number of iterations taken
// fret: is the returned function value at Vec_IO_DP (p)
// DP func: The routine "linmin" is used. linmin:Given as input the vectors P (point) and n (direction), and the function f, 
//          find the scalar lamda that minimizes f(P+lamda n ) Replace P by P + lamda n. Replace n by lamda n.
// minimize_target_unscented_kalman_parameters_1_dim
{
	const int ITMAX=200;
	const DP TINY=1.0e-25;
	int i,j,ibig;
	DP del,fp,fptt,t;

	int n=p.size();
	Vec_DP pt(n),ptt(n),xit(n);
	fret=func(p);
	for (j=0;j<n;j++) pt[j]=p[j];    //save the initial point
	for (iter=0;;++iter) {
		fp=fret;
		ibig=0;
		del=0.0;   //will be the biggest function decrease
		for (i=0;i<n;i++) {   //in each iteration, loop over all directions in the set
			for (j=0;j<n;j++) xit[j]=xi[j][i];    //copy the direction
			fptt=fret;
			//NR::void linmin(Vec_IO_DP &p, Vec_IO_DP &xi, DP &fret, DP func(Vec_I_DP &)); //in nr.h 
			linmin(p,xit,fret,func);   //minimize along it
			if (fptt-fret > del) {     // and record it if it is the largest decrease so far.
				del=fptt-fret;
				ibig=i+1;
			}
		}
		if (2.0*(fp-fret) <= ftol*(fabs(fp)+fabs(fret))+TINY) {
			return;    //Termination criterion
		}
		if (iter == ITMAX) nrerror("powell exceeding maximum iterations.");
		for (j=0;j<n;j++) {    //Construct the extrapolated point and the average direction moved. save
			ptt[j]=2.0*p[j]-pt[j]; //the old starting point
			xit[j]=p[j]-pt[j];
			pt[j]=p[j];
		}
		fptt=func(ptt);    //Function value at extrapolated point
		if (fptt < fp) {
			t=2.0*(fp-2.0*fret+fptt)*SQR(fp-fret-del)-del*SQR(fp-fptt);
			if (t < 0.0) {
				linmin(p,xit,fret,func);    //Move to the minimum of the new direction, and 
				for (j=0;j<n;j++) {         //save the new direction  
					xi[j][ibig-1]=xi[j][n-1];
					xi[j][n-1]=xit[j];
				}
			}
		}     //Back for another iteration
	}
}


