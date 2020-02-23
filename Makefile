TARGET		= pfw

NVCC		= nvcc

CUFLAGS		= -g

SRC		= $(wildcard *.cu)
INC		= $(wildcard *.cuh)

OBJ		= $(patsubst %.cu,%.o, $(SRC))

$(TARGET): $(OBJ) $(INC)
	$(NVCC) $(CUFLAGS) $(OBJ) -o $@

%.o:%.cu
	$(NVCC) $(CUFLAGS) $^ -c $@

clean:
	rm -rf $(OBJ) $(TARGET)


