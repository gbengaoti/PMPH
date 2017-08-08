CXX        = g++
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL
CXXFLAGS   = -fopenmp -O3 -DWITH_FLOATS=0

INCLUDES    += -I ../include
GPU_OPTS   = -D lgWARP=5

SOURCES_CPP =ProjectMain.cpp ProjHelperFun.cpp ProjCoreOrig.cpp 
HELPERS     =ProhHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h
OBJECTS     =ProjectMain.o ProjHelperFun.o ProjCoreOrig.o
ifeq ($(OS),Windows_NT)
EXECUTABLE  =runproject.exe
else
EXECUTABLE  =runproject
endif

default: cpu

.cpp.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<


cpu: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) $(OBJECTS)


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
ifeq ($(OS),Windows_NT)
	del /S Debug.txt $(EXECUTABLE) $(OBJECTS) >nul 2>&1
else
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)
endif
