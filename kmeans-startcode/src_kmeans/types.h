#pragma once
#include <vector>
#include <iostream>

typedef std::vector<double> Point;

inline void printPoint(Point p){
    for (double d: p){
        std::cout << d << ",";
    }
    std::cout << "\n";
}