## http://stackoverflow.com/questions/16931770/makefile4-missing-separator-stop

#alias the g++ command so that we don't have to sepecify
#compile=g++ -c -o3


# Add source files here
EXECUTABLE := oclThreeDimArray
# C/C++ source files (compiled with gcc / c++)
CCFILES	:= oclThreeDimArray.cpp 

include ../../common/common_opencl.mk

################################################################################
# Rules and targets





#all: final_ekf

#final_ekf: dir bin/filters.o bin/powell.o bin/filter_utils.o bin/ran2.o bin/bessik.o bin/final_ekf.o
#	g++ bin/*.o -o bin/final_ekf	

#bin/final_ekf.o: final_ekf.cpp
#	${compile} final_ekf.cpp -o bin/final_ekf.o

## related libraries
#bin/filters.o: filters.cpp
#	${compile} filters.cpp -o bin/filters.o

#bin/powell.o: bin/f1dim.o bin/linmin.o bin/mnbrak.o bin/brent.o bin/choldc.o recipes/powell.cpp
#	${compile} recipes/powell.cpp -o bin/powell.o

#bin/ran2.o: recipes/ran2.cpp
#	${compile} recipes/ran2.cpp -o bin/ran2.o

#bin/f1dim.o: recipes/f1dim.cpp
#	${compile} recipes/f1dim.cpp -o bin/f1dim.o

#bin/linmin.o: recipes/linmin.cpp
#	${compile} recipes/linmin.cpp -o bin/linmin.o

#bin/mnbrak.o: recipes/mnbrak.cpp
#	${compile} recipes/mnbrak.cpp -o bin/mnbrak.o

#bin/brent.o: recipes/brent.cpp
#	${compile} recipes/brent.cpp -o bin/brent.o

#bin/choldc.o: recipes/choldc.cpp
#	${compile} recipes/choldc.cpp -o bin/choldc.o

#bin/filter_utils.o: bin/choldc.o bin/chsone.o bin/moment.o filter_utils.h filter_utils.cpp
#	${compile} filter_utils.cpp -o bin/filter_utils.o

#bin/chsone.o: bin/gammq.o recipes/chsone.cpp 
#	${compile} recipes/chsone.cpp -o bin/chsone.o

#bin/moment.o: recipes/moment.cpp
#	${compile} recipes/moment.cpp -o bin/moment.o

#bin/gammq.o: bin/gser.o bin/gcf.o recipes/gammq.cpp
#	${compile} recipes/gammq.cpp -o bin/gammq.o

#bin/gser.o: bin/gammln.o recipes/gser.cpp
#	${compile} recipes/gser.cpp -o bin/gser.o

#bin/gcf.o: bin/gammln.o recipes/gcf.cpp
#	${compile} recipes/gcf.cpp -o bin/gcf.o

#bin/gammln.o: recipes/gammln.cpp
#	${compile} recipes/gammln.cpp -o bin/gammln.o

#bin/chebev.o: recipes/chebev.cpp
#	${compile} recipes/chebev.cpp -o bin/chebev.o


#bin/beschb.o: recipes/beschb.cpp bin/chebev.o
#	${compile} recipes/beschb.cpp -o bin/beschb.o

#bin/bessik.o: recipes/bessik.cpp bin/beschb.o
#	${compile} recipes/bessik.cpp -o bin/bessik.o



#dir:
#	mkdir -p bin
#clean: 
#	rm -v ./bin/*
#ekf:
#	g++ ekf.cpp filters.cpp ./recipes/powell.cpp ./recipes/f1dim.cpp ./recipes/linmin.cpp ./recipes/mnbrak.cpp ./recipes/brent.cpp -o ekf	
