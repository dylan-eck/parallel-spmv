out: main.o
	g++ -pthread -fopenmp main.o -o out 

main.o: main.cpp
	g++ -I/usr/include/eigen3 -O3 -c main.cpp

run:
	./out

clean:
	rm *.o out