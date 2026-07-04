# Makefile
CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -pthread -Wall -Wextra
LDFLAGS = -lpthread -lm -lstdc++ -lz

TARGET = nova

SOURCES = main.cpp \
    main/remaining/gguf.cpp \
    main/remaining/st.cpp \
    main/remaining/cache.cpp \
    main/remaining/pa.cpp \
    main/remaining/fa.cpp \
    main/remaining/moe.cpp \
    main/remaining/s.cpp \
    main/remaining/i.cpp \
    main/remaining/train.cpp \
    main/remaining/op.cpp \
    main/remaining/l.cpp \
    main/remaining/tll.cpp \
    main/remaining/train.cpp \
    main/remaining/llama.cpp \
    main/remaining/http.cpp \
    main/remaining/cli.cpp \
    main/remaining/download.cpp \
    main/remaining/conv.cpp \
    main/remaining/q.cpp \
    main/remaining/p.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

install:
	cp $(TARGET) /usr/local/bin/

run: $(TARGET)
	./$(TARGET) server

.PHONY: all clean install run