cuda := nvcc

all :
	@$(cuda) parallel_reduction.cu -o parallel_reduction

run1 :
	@./parallel_reduction 1 32

run2 :
	@./parallel_reduction 2 32

prof :
	@sudo /usr/local/cuda/bin/ncu --set full ./parallel_reduction 1 32

prof1 : 
	@sudo /usr/local/cuda/bin/ncu --launch-skip 4 --launch-count 1 ./parallel_reduction 1 32

prof2 : 
	@sudo /usr/local/cuda/bin/ncu --launch-skip 4 --launch-count 1 ./parallel_reduction 2 32

clean :
	@rm parallel_reduction

.PHONY : all clean
