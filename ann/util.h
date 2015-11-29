#ifndef UTIL__H
#define UTIL__H

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <iomanip>
#include <iostream>

class Util{
public:
    static std::vector<std::string> splitv(const std::string &s, char delim, std::vector<std::string> &elems);
    static std::vector<std::string> split(const std::string &s, char delim);
    static std::string trim(std::string &s);
	static std::vector<double> split_numeric(const std::string &s, char delim);
private:

};
#endif
