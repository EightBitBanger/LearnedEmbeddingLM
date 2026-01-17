#include <unordered_map>
#include <unordered_set>
#include <string>

#include "string.h"

std::vector<std::string> StringExplode(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < s.size(); ++i) {
        char ch = s[i];
        if (ch == delim) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else if (ch == '\n' || ch == '\r' || ch == '\t') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

void StringCaseUpper(std::string& str) {
    bool first_alpha_done = false;
    
    for (size_t i = 0, n = str.size(); i < n; ++i) {
        unsigned char uc = (unsigned char)str[i];
        
        if (!first_alpha_done && std::isalpha(uc)) {
            str[i] = (char)std::toupper(uc);
            first_alpha_done = true;
        } else if (first_alpha_done && std::isalpha(uc)) {
            str[i] = (char)std::tolower(uc);
        }
    }
}

void StringCaseLower(std::string& str) {
    bool first_alpha_done = false;
    
    for (size_t i = 0, n = str.size(); i < n; ++i) {
        unsigned char uc = (unsigned char)str[i];
        
        if (!first_alpha_done && std::isalpha(uc)) {
            str[i] = (char)std::tolower(uc);
            first_alpha_done = true;
        } else if (first_alpha_done && std::isalpha(uc)) {
            str[i] = (char)std::tolower(uc);
        }
    }
}

void StringCaseLowerAll(std::string& str) {
    for (size_t i = 0, n = str.size(); i < n; ++i) {
        unsigned char uc = (unsigned char)str[i];
        if (std::isalpha(uc)) 
            str[i] = (char)std::tolower(uc);
    }
}

bool StringCheckIsEndPunctuation(const std::string& s) {
    static const std::unordered_set<std::string> P = {".", "!", "?"};
    return P.count(s) > 0;
}


bool StringCheckIsWordish(const std::string& t) {
    if (t.empty()) return false;
    
    bool has_alpha = false;
    for (size_t i = 0; i < t.size(); i++) {
        unsigned char uc = (unsigned char)t[i];
        if (std::isalpha(uc)) {has_alpha = true; continue;}
        if (std::isdigit(uc)) continue;
        if (uc == '\'' || uc == '-' || uc == '_') continue;
        return false;
    }
    return has_alpha;
}

bool IsNoSpaceBeforePunct(const std::string& w) {
    return (w == "." || w == "," || w == "!" || w == "?" ||
            w == ":" || w == ";" || w == ")" || w == "]" || w == "}");
}

bool IsOpenBracket(const std::string& w) {
    return (w == "(" || w == "[" || w == "{");
}

bool IsSentenceEnd(const std::string& w) {
    return (w == "." || w == "!" || w == "?");
}

bool IsSkippableLeadingToken(const std::string& w) {
    return (w == "." || w == "!" || w == "?" || w == "," ||
            w == ":" || w == ";");
}
