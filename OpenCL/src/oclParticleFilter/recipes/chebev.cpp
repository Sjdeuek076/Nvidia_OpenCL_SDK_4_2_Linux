#include "nr.h"

DP NR::chebev(const DP a, const DP b, Vec_I_DP &c, const int m, const DP x)
//DP NR::chebev(DP a,  DP b, Vec_I_DP &c,  int m,  DP x)
{
	DP d=0.0,dd=0.0,sv,y,y2;
	int j;

	if ((x-a)*(x-b) > 0.0)
		nrerror("x not in range in routine chebev");
	y2=2.0*(y=(2.0*x-a-b)/(b-a));
	for (j=m-1;j>0;j--) {
		sv=d;
		d=y2*d-dd+c[j];
		dd=sv;
	}
	return y*d-dd+0.5*c[0];
}
