CXX := mpicxx
CXXFLAGS := -O3

all: matmul

matmul: main.o matrix.o
	$(CXX) $(CXXFLAGS) main.o matrix.o -o matmul

%.o: %.cc
	${CXX} ${CXXFLAGS} -c $^ -o $@

.PHONY: clean test
clean:
	rm -rf matmul *.o *~
