# cfiles, project_name
CFILES := $(wildcard *.c)
PROJECT_NAME := net

# c
Q := @
CC = clang
CFLAGS = -Wall -Werror -g -DDEBUG -O3

.PHONY: all
all: $(PROJECT_NAME)
$(PROJECT_NAME): $(CFILES)
	@ echo "Compile and Link" $@
	$(Q)$(CC) $(CFLAGS) -o $@ $^

.PHONY:clean
clean:
	@ rm -f $(PROJECT_NAME)
	@ rm -rf $(PROJECT_NAME).dSYM

.PHONY:test
test: $(PROJECT_NAME)
	@ echo "test" $^
	@ ./$(PROJECT_NAME)
