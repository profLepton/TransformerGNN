OBJECTS = *.pt
LOCATIONS = $(shell find . -name '*.pt')
clean:
	rm -rf $(LOCATIONS)