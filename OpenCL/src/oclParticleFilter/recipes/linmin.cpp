#include "nr.h"

int ncom;
DP (*nrfunc)(Vec_I_DP &);
Vec_DP *pcom_p,*xicom_p;

void NR::linmin(Vec_IO_DP &p, Vec_IO_DP &xi, DP &fret, DP func(Vec_I_DP &))
/* p.419 Given an n-dimensional point p[1..n] and n-dimensional direction xi[1..n], moves
 * and resets p to where the function func(p) takes on a minimum along the direction xi from p,
 * and replaces xi by the actual vector displacement that p was moved. Also returns as fret the 
 * value of func at the returned location p. This is actually all accomplished by calling the
 * routines mnbrak and brent.
 */
{
	int j;
	const DP TOL=1.0e-8;	//Tolerance passed to brent
	DP xx,xmin,fx,fb,fa,bx,ax;

	int n=p.size();
	ncom=n;		//define the global variables
	pcom_p=new Vec_DP(n);
	xicom_p=new Vec_DP(n);
	nrfunc=func;
	Vec_DP &pcom=*pcom_p,&xicom=*xicom_p;
	for (j=0;j<n;j++) {
		pcom[j]=p[j];
		xicom[j]=xi[j];
	}
	ax=0.0;		//Initial guess for brackets
	xx=1.0;
	mnbrak(ax,xx,bx,fa,fx,fb,f1dim);
	fret=brent(ax,xx,bx,f1dim,TOL,xmin);
	for (j=0;j< n;j++) {	//Construct the vector results to return.
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	delete xicom_p;
	delete pcom_p;
}
