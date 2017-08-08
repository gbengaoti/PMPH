# NOTE: This makefile assumes that:
# 1) ProjectMain.cpp has been renamed to ProjectMain.cu
# 2) ProjectMain.cu include of "OpenmpUtil.h" has been changed to "CudaUtilProj.cu.h"
# 3) The line "const int Ps = get_CPU_num_threads();" has been disabled
# To compile on Windows, some modifications to ParseInput are needed as sys/time.h is unix.
# It should work for all data sets, but the large data set takes a long time.
CXX        = nvcc
CXXFLAGS   = -DWITH_FLOATS=0

INCLUDES    += -I ../include

SOURCES_CPP =ProjectMain.cu ProjHelperFun.cpp ProjCoreOrigCuda.cu 
HELPERS     =ProhHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/CudaUtilProj.cu.h
EXECUTABLE  =runproject


default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) $(SOURCES_CPP)


run_small: $(EXECUTABLE)
ifeq ($(OS),Windows_NT)
	type ..\Data\Small\input.data ..\Data\Small\output.data | $(EXECUTABLE)
else
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt
endif

run_medium: $(EXECUTABLE)
ifeq ($(OS),Windows_NT)
	type ..\Data\Medium\input.data ..\Data\Medium\output.data | $(EXECUTABLE)
else
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt
endif

run_large: $(EXECUTABLE)
ifeq ($(OS),Windows_NT)
	type ..\Data\Large\input.data ..\Data\Large\output.data | $(EXECUTABLE)
else
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt
endif

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)

