#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "gsa.h"
#include "languagemodel.h"

extern LanguageModel model;

GlobalSemanticAlignment::GlobalSemanticAlignment(Tokenizer* tokPtr, SamplerSystem* samplerPtr, AttentionSystem* attentionPtr) : 
    tok(tokPtr),
    sampler(samplerPtr),
    attention(attentionPtr) {
    
    RefreshLexicons();
}

float GlobalSemanticAlignment::GetQuestionScore(const std::vector<int>& context) {
    if (context.empty())
        return 0.0f;

    const int tokPeriod  = model.tok.wordToToken.count(".") ? model.tok.wordToToken["."] : -1;
    const int tokQMark   = model.tok.wordToToken.count("?") ? model.tok.wordToToken["?"] : -1;
    const int tokExclaim = model.tok.wordToToken.count("!") ? model.tok.wordToToken["!"] : -1;

    if (tokQMark != -1 && context.back() == tokQMark)
        return 1.0f;

    auto TokenInList = [&](int t, const std::vector<int>& list) -> bool {
        for (unsigned int i = 0; i < list.size(); ++i) {
            if (list[i] == t)
                return true;
        }
        return false;
    };

    int sentenceStart = 0;
    for (int i = (int)context.size() - 1; i >= 0; --i) {
        const int t = context[(unsigned int)i];
        if ((tokPeriod  != -1 && t == tokPeriod) ||
            (tokExclaim != -1 && t == tokExclaim) ||
            (tokQMark   != -1 && t == tokQMark)) {
            sentenceStart = i + 1;
            break;
        }
    }

    auto IsSkippablePunct = [&](int t) -> bool {
        static const char* kPunct[] = { "\"", "'", "(", ")", "[", "]", "{", "}", ",", ";", ":", "-", "—" };
        for (unsigned int i = 0; i < (unsigned int)(sizeof(kPunct) / sizeof(kPunct[0])); ++i) {
            std::unordered_map<std::string, int>::const_iterator it = model.tok.wordToToken.find(kPunct[i]);
            if (it != model.tok.wordToToken.end() && t == it->second)
                return true;
        }
        return false;
    };

    int firstWordTok = -1;
    for (unsigned int i = (unsigned int)sentenceStart; i < context.size(); ++i) {
        const int t = context[i];
        if (t == tokPeriod || t == tokExclaim || t == tokQMark)
            continue;
        if (IsSkippablePunct(t))
            continue;
        firstWordTok = t;
        break;
    }

    if (firstWordTok == -1)
        return 0.0f;

    float score = 0.0f;

    if (TokenInList(firstWordTok, questionWords))
        score = std::max(score, 0.90f);

    if (TokenInList(firstWordTok, beAux) || TokenInList(firstWordTok, haveAux) || TokenInList(firstWordTok, doAux))
        score = std::max(score, 0.70f);

    if (tokQMark != -1) {
        const int tailScan = 12;
        int start = (int)context.size() - tailScan;
        if (start < 0) start = 0;
        for (int i = (int)context.size() - 1; i >= start; --i) {
            if (context[(unsigned int)i] == tokQMark) {
                score = std::max(score, 0.95f);
                break;
            }
        }
    }

    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;
    return score;
}

std::vector<int> GlobalSemanticAlignment::GetQuestionSubject(const std::vector<int>& context, unsigned int maxTokens) {
    std::vector<int> out;
    if (context.empty() || maxTokens == 0)
        return out;

    const int tokPeriod  = model.tok.wordToToken.count(".") ? model.tok.wordToToken["."] : -1;
    const int tokQMark   = model.tok.wordToToken.count("?") ? model.tok.wordToToken["?"] : -1;
    const int tokExclaim = model.tok.wordToToken.count("!") ? model.tok.wordToToken["!"] : -1;

    int sentenceStart = 0;
    int sentenceEnd = (int)context.size();

    while (sentenceEnd > 0) {
        const int t = context[(unsigned int)(sentenceEnd - 1)];
        if ((tokPeriod  != -1 && t == tokPeriod) ||
            (tokExclaim != -1 && t == tokExclaim) ||
            (tokQMark   != -1 && t == tokQMark)) {
            sentenceEnd--;
            continue;
        }
        break;
    }

    for (int i = sentenceEnd - 1; i >= 0; --i) {
        const int t = context[(unsigned int)i];
        if ((tokPeriod  != -1 && t == tokPeriod) ||
            (tokExclaim != -1 && t == tokExclaim) ||
            (tokQMark   != -1 && t == tokQMark)) {
            sentenceStart = i + 1;
            break;
        }
    }

    auto TokenInList = [&](int t, const std::vector<int>& list) -> bool {
        for (unsigned int i = 0; i < list.size(); ++i) {
            if (list[i] == t)
                return true;
        }
        return false;
    };

    auto IsSkippablePunct = [&](int t) -> bool {
        static const char* kPunct[] = { "\"", "'", "(", ")", "[", "]", "{", "}", ",", ";", ":", "-", "—" };
        for (unsigned int i = 0; i < (unsigned int)(sizeof(kPunct) / sizeof(kPunct[0])); ++i) {
            std::unordered_map<std::string, int>::const_iterator it = model.tok.wordToToken.find(kPunct[i]);
            if (it != model.tok.wordToToken.end() && t == it->second)
                return true;
        }
        return false;
    };

    static std::vector<int> extraStop;
    static bool extraStopInit = false;
    if (!extraStopInit) {
        extraStopInit = true;
        const char* kStop[] = {
            "a","an","the","and","or","but",
            "this","that","these","those",
            "can","could","would","should","will","may","might","must",
            "please"
        };
        for (unsigned int i = 0; i < (unsigned int)(sizeof(kStop) / sizeof(kStop[0])); ++i) {
            std::unordered_map<std::string,int>::const_iterator it = model.tok.wordToToken.find(kStop[i]);
            if (it != model.tok.wordToToken.end())
                extraStop.push_back(it->second);
        }
    }

    for (int i = sentenceStart; i < sentenceEnd; ++i) {
        const int t = context[(unsigned int)i];

        if (t == tokPeriod || t == tokExclaim || t == tokQMark)
            continue;
        if (IsSkippablePunct(t))
            continue;

        if (TokenInList(t, questionWords))      continue;
        if (TokenInList(t, beAux))              continue;
        if (TokenInList(t, haveAux))            continue;
        if (TokenInList(t, doAux))              continue;
        if (TokenInList(t, negationWords))      continue;
        if (TokenInList(t, prepositionWords))   continue;
        if (TokenInList(t, intensityUpWords))   continue;
        if (TokenInList(t, intensityDownWords)) continue;
        if (TokenInList(t, pronounWords))       continue;
        if (TokenInList(t, extraStop))          continue;

        if (!out.empty() && out.back() == t)
            continue;

        out.push_back(t);
        if (out.size() >= maxTokens)
            break;
    }

    return out;
}


int GlobalSemanticAlignment::SampleAligned(const std::vector<int>& context, const std::vector<std::vector<int>>& focus, const SamplerParameters& params) {
    //TokenDistribution dist = sampler->SampleNextTokenDistribution(context, focus, params, 40);
    //for (unsigned int i=0; i < dist.tokens.size(); i++) 
    //    std::cout << dist.weights[i] << "    " << model.tok.tokenToWord[dist.tokens[i]] << "\n";
    
    int token = sampler->SampleNextToken(context, focus, params);
    
    return token;
}

void GlobalSemanticAlignment::RefreshLexicons() {
    questionWords.clear();
    negationWords.clear();
    futureMarkers.clear();
    pastMarkers.clear();
    beAux.clear();
    haveAux.clear();
    doAux.clear();
    prepositionWords.clear();
    intensityUpWords.clear();
    intensityDownWords.clear();
    pronounWords.clear();
    
    // Question words
    ResolveWordList(std::vector<std::string>{"who", "what", "when", "where", "why", "how"}, questionWords);
    
    // Negation
    ResolveWordList(std::vector<std::string>{"not", "never", "no", "n't", "cannot", "can't", "dont", "don't"}, negationWords);
    
    // Tense markers
    ResolveWordList(std::vector<std::string>{"will", "shall", "gonna", "going"}, futureMarkers);
    ResolveWordList(std::vector<std::string>{"was", "were", "had", "did", "yesterday", "ago"}, pastMarkers);
    
    // Auxiliaries
    ResolveWordList(std::vector<std::string>{"am", "is", "are", "was", "were", "been", "being", "be"}, beAux);
    ResolveWordList(std::vector<std::string>{"have", "has", "had"}, haveAux);
    ResolveWordList(std::vector<std::string>{"do", "does", "did"}, doAux);
    
    // Prepositions
    ResolveWordList(std::vector<std::string>{
        "by", "in", "on", "at", "from", "to", "with", "for", "of", 
        "into", "over", "under", "between", "through", "during", "before", "after"}, prepositionWords);
    
    // Intensity cues
    ResolveWordList(std::vector<std::string>{"very", "really", "so", "too", "extremely", "highly"}, intensityUpWords);
    ResolveWordList(std::vector<std::string>{"barely", "hardly", "scarcely", "slightly"}, intensityDownWords);
    
    // Pronouns
    ResolveWordList(std::vector<std::string>{
        "i", "me", "my", "mine",    "you", "your", "yours",
        "he", "him", "his",         "she", "her", "hers",
        "we", "us", "our", "ours",  "they", "them", "their", "theirs",
        "it", "its"}, pronounWords);
}

void GlobalSemanticAlignment::ResolveWordList(const std::vector<std::string>& words, std::vector<int>& outTokens) {
    outTokens.clear();
    
    if (tok == NULL) {
        return;
    }
    
    for (unsigned int i = 0; i < words.size(); i++) {
        const std::string& word = words[i];
        if (!tok->CheckWordExists(word)) {
            continue;
        }
        outTokens.push_back(tok->wordToToken[word]);
    }
}
