# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -lcusparse
CXX = g++
CXXFLAGS = -std=c++11

# Target executable
TARGET = build/test

# Source files
CU_SOURCES = src/test.cu 
CPP_SOURCES = src/MatrixParser.cpp src/MatrixHelper.cpp

# Object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Rules
all: $(TARGET)

$(TARGET): $(CU_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^
	rm -f $(CU_OBJECTS) $(CPP_OBJECTS)

%.o: %.cu
	$(NVCC) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CU_OBJECTS) $(CPP_OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean
