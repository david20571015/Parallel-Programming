CXX := g++
OBJ := pi.o
OUT := pi.out
CXXFLAGS := -O3 -std=c++17 -Wall -pthread -mavx -mavx2 -march=native

ifeq (/usr/bin/g++-10,$(wildcard /usr/bin/g++-10*))
    CXX=g++-10
endif

all: ${OBJ}
	${CXX} $(CXXFLAGS) ${OBJ} -o ${OUT}

%.o: %.cpp
	${CXX} $(CXXFLAGS) -c $^ -o $@

.PHONY:clean
clean:
	rm ${OBJ} ${OUT}