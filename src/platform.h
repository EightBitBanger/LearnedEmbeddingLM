#ifndef _PLATFORM__
#define _PLATFORM__

#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

bool KeyPressedNonBlocking();
int ReadKeyNonBlocking();

int RandomRange(int min, int max);

bool FileTextLoad(const std::string& filename, std::string& out);

bool FileExists(const std::string& filename);

bool DirectoryExists(const std::string& path);

std::string FloatToString(float value);
float StringToFloat(const std::string& value);
int StringToInt(const std::string& value);
std::string IntToString(int value);

#endif
