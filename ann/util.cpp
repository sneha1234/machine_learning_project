
#include "util.h"
//Functions taken from stack overflow
std::vector<std::string> Util::splitv(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<double> Util::split_numeric(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
	std::vector<double> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(std::stof(item));
    }
    return elems;
}

std::vector<std::string> Util::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    splitv(s, delim, elems);
    return elems;
}

// trim from start
static inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
std::string Util::trim(std::string &s) {
        return ltrim(rtrim(s));
}
