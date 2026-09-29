CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

SRCS = main.cpp \
       Matrix.cpp \
       ParameterEstimator.cpp \
       ConjugateGradientSolver.cpp \
       LinearSystemSolver.cpp \
       PortfolioOptimiser.cpp \
       Backtester.cpp \
       csv_reader.cpp

OBJS   = $(SRCS:.cpp=.o)
TARGET = markowitz

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) *.csv

.PHONY: all clean
