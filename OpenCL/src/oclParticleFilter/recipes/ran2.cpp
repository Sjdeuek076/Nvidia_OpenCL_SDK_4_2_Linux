//nr p.275 (pdf 299)
#include "nr.h"

DP NR::ran2(int &idum)
{
	const int IM1=2147483563,IM2=2147483399;
	const int IA1=40014,IA2=40692,IQ1=53668,IQ2=52774;
	const int IR1=12211,IR2=3791,NTAB=32,IMM1=IM1-1;
	const int NDIV=1+IMM1/NTAB;
	const DP EPS=3.0e-16,RNMX=1.0-EPS,AM=1.0/DP(IM1);	
	static long idum2=123456789,iy=0;
	static Vec_INT iv(NTAB);
	int j,k;
	DP temp;

	if (idum <= 0) {        //initialize
		idum=(idum==0 ? 1 : -idum);     //Be sure to prevent idum=0
		idum2=idum;
		for (j=NTAB+7;j>=0;j--) {       //Load the shuffle table (after 8 warm-ups)
			k=idum/IQ1;
			idum=IA1*(idum-k*IQ1)-k*IR1;
			if (idum < 0) idum += IM1;
			if (j < NTAB) iv[j] = idum;
		}
		iy=iv[0];
	}
	k=idum/IQ1;         //Start here when not initializing
	idum=IA1*(idum-k*IQ1)-k*IR1;    //compute idum=(IA1*idum) % IM1 without 
	if (idum < 0) idum += IM1;      //overflows by Schrage's method
	k=idum2/IQ2;
	idum2=IA2*(idum2-k*IQ2)-k*IR2;  //Compute idum2=(IA2*idum) % IM2 likewise
	if (idum2 < 0) idum2 += IM2;
	j=iy/NDIV;          //will be in the range 0..NTAB-1
	iy=iv[j]-idum2;     //Here idum is shuffled, idum and idum2 are 
	iv[j] = idum;       //combined to generate output
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) return RNMX;       //Because users don't expect endpoint values
	else return temp;
}
