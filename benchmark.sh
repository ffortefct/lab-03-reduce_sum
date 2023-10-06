#!/bin/sh

SZ=252
RUNS=3
PROGS="sum sum-omp sum-cuda"

for p in $PROGS; do
	echo "running $p..."

	file="$p.csv"

	rm -f $file

	echo "optimization,runtime,array_size,init_wall,init_cpu,ref_wall,ref_cpu" >>$file

	for optimization in {0..3}; do
		for runtime in $(seq 1 $RUNS); do
			out=$(make -e OPT=$optimization clean $p && ./$p $SZ)
			[[ "$?" -ne 0 ]] && exit 1
			parsed=$(echo $out | awk -F "-o $p" '{print $2}')
			array_size=$(echo $parsed | cut -d " " -f 6)
			init_wall=$(echo $parsed | cut -d " " -f 9 | cut -d "=" -f 2)
			init_cpu=$(echo $parsed | cut -d " " -f 11 | cut -d "=" -f 2)
			ref_wall=$(echo $parsed | cut -d " " -f 17 | cut -d "=" -f 2)
			ref_cpu=$(echo $parsed | cut -d " " -f 19 | cut -d "=" -f 2)
			echo "$optimization,$runtime,$array_size,$init_wall,$init_cpu,$ref_wall,$ref_cpu" >>$file
		done
	done
done
