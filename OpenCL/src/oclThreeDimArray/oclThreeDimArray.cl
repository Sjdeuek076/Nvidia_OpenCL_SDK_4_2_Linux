__kernel void ThreeDimArray(const __global float * const input1, 
			       const __global float * const input2,	
			       __global float * const output1,		
			       __global float * const output2,		
			       __global float * const output3){		
 const int x = get_global_id(0);					
 const int y = get_global_id(1);					
 const int z = get_global_id(2);					
 const int max_x = get_global_size(0);				
 const int max_y = get_global_size(1);				
 const int max_z = get_global_size(2);				
 const int idx = x * max_y * max_z + y * max_z + z;			
 output1[idx] = x;							
 output2[idx] = y;							
 output3[idx] = z;							
 //Uncommnet the next line if you want to see the input values used. \n"
 //output1[idx] = input1[idx] + input2[idx];				\n"
  }
