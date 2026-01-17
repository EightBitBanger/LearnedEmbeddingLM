#ifndef _STRINGS__
#define _STRINGS__

#include <string>
#include <vector>

std::vector<std::string> StringExplode(const std::string& s, char delim);
void StringCaseUpper(std::string& str);
void StringCaseLower(std::string& str);
void StringCaseLowerAll(std::string& str);
bool StringCheckIsEndPunctuation(const std::string& s);
bool StringCheckIsWordish(const std::string& t);

bool IsNoSpaceBeforePunct(const std::string& w);
bool IsOpenBracket(const std::string& w);
bool IsSentenceEnd(const std::string& w);
bool IsSkippableLeadingToken(const std::string& w);

#endif
