cuda := nvcc

all :
	@$(cuda) -w softmax.cu -o softmax

run1 :
	@./softmax 1

run2 :
	@./softmax 2

run3 :
	@./softmax 3

clean :
	@rm softmax

.PHONY : all clean
