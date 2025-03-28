cuda := nvcc

all :
	@$(cuda) -w layernorm.cu -o layernorm

run1 :
	@./layernorm 1

run2 :
	@./layernorm 2

run3 :
	@./layernorm 3

run4 :
	@./layernorm 4

run5 :
	@./layernorm 5

run6 :
	@./layernorm 6

clean :
	@rm layernorm

.PHONY : all clean
