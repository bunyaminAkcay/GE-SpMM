# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -lcusparse
CXX = g++
CXXFLAGS = -std=c++11

# Target executable
TARGET1 = build/test
TARGET2 = build/SimpleCsrSpMMTest
TARGET3 = build/CRCSpMMTest
TARGET4 = build/CRC-CWM-SpMM-Test

# Source files
CU_SOURCES1 = src/test/test.cu 
CU_SOURCES_CSR = src/test/SimpleCsrSpMMTest.cu
CU_SOURCES_CRC = src/test/CRCSpMMTest.cu
CU_SOURCES_CRC_CWM = src/test/CRC-CWM-SpMM-Test.cu


CPP_SOURCES = src/MatrixParser.cpp src/MatrixHelper.cpp

# Object files
CU_OBJECTS1 = $(CU_SOURCES1:.cu=.o)
CU_OBJECTS_CSR = $(CU_SOURCES_CSR:.cu=.o)
CU_OBJECTS_CRC = $(CU_SOURCES_CRC:.cu=.o)
CU_OBJECTS_CRC_CWM = $(CU_SOURCES_CRC_CWM:.cu=.o)

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Rules
all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)

$(TARGET1): $(CU_OBJECTS1) $(CPP_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(TARGET2): $(CU_OBJECTS_CSR) $(CPP_OBJECTS)
	$(NVCC) -o $@ $^
	

$(TARGET3): $(CU_OBJECTS_CRC) $(CPP_OBJECTS)
	$(NVCC) -o $@ $^

$(TARGET4): $(CU_OBJECTS_CRC_CWM) $(CPP_OBJECTS)
	$(NVCC) -o $@ $^
	rm -f $(CU_OBJECTS1) $(CU_OBJECTS_CSR) $(CU_OBJECTS_CRC) $(CU_OBJECTS_CRC_CWM) $(CPP_OBJECTS)

%.o: %.cu
	$(NVCC) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CU_OBJECTS1) $(CU_OBJECTS_CSR) $(CU_OBJECTS_CRC) $(CU_OBJECTS_CRC_CWM) $(CPP_OBJECTS) $(TARGET1)$(TARGET2)$(TARGET3)$(TARGET4)

# Phony targets
.PHONY: all clean