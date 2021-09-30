CC := gcc
obj := pi.o
out := pi.out

all: ${obj}
	${CC} -o ${out} ${obj}
%.o: %.c
	${CC} -c $^ -o $@

.PHONY:clean
clean:
	rm ${obj} ${out}