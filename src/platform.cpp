#include "platform.h"

#include <conio.h>

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <random>

std::mt19937 rng(
    static_cast<unsigned long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    )
);

bool KeyPressedNonBlocking() {
    return _kbhit() != 0;
}

int ReadKeyNonBlocking() {
    return _getch();
}

int RandomRange(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}


bool FileTextLoad(const std::string& filename, std::string& out) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

bool FileExists(const std::string& filename) {
    std::ifstream stream(filename);
    return stream.is_open();
}

bool DirectoryExists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return false;
    return S_ISDIR(st.st_mode);
}

std::vector<std::string> ListDirectoryFiles(const std::string& path) {
    std::vector<std::string> result;
    DIR* dir = opendir(path.c_str());
    if (!dir)
        return result;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        const char* name = entry->d_name;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
            continue;
        result.emplace_back(name);
    }
    closedir(dir);
    return result;
}

std::string FloatToString(float value) {
    std::stringstream sstream;
    sstream << value;
    return sstream.str();
}

float StringToFloat(const std::string& value) {
    float output;
    std::stringstream(value) >> output;
    return output;
}

int StringToInt(const std::string& value) {
    int output;
    std::stringstream(value) >> output;
    return output;
}

std::string IntToString(int value) {
    std::stringstream sstream;
    sstream << value;
    return sstream.str();
}
