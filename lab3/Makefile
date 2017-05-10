all:
	g++ -std=c++11 -Wno-unused-result -c pgm.cpp
	nvcc -I../utils -std=c++11 -O2 -o lab3 lab3.cu main.cu pgm.o

clean:
	rm -f lab3 pgm.o

test:
	./lab3 img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm
	convert output.ppm results/output.png
