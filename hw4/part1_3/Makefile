CC := mpicxx
RUN := mpirun
NP := 2

all: ping_pong

ping_pong: ping_pong.c
	$(CC) ping_pong.c -o ping_pong

PHONY: run_inter run_intra

run_inter: ping_pong hosts_inter
	$(RUN) -np $(NP) -hostfile hosts_inter ping_pong

run_intra: ping_pong hosts_intra
	$(RUN) -np $(NP) -hostfile hosts_intra ping_pong