cuda := nvcc

all :
	@$(cuda) -w softmax.cu -o softmax

run1 :
	@./softmax 1

run2 :
	@./softmax 2

run3 :
	@./softmax 3

run4 :
	@./softmax 4

run5 :
	@./softmax 5

run6 :
	@./softmax 6

clean :
	@rm softmax

.PHONY : all clean
