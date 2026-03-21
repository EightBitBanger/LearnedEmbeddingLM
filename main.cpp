#include <string>
#include <vector>
#include <unordered_map>

#include <sstream>
#include <iostream>

#include "repl.h"
#include "string.h"
#include "platform.h"

#include "context.h"
#include "tokenizer.h"
#include "attention.h"
#include "languagemodel.h"
#include "sampler.h"
#include "rem.h"

ReplCommandConsole console;

Tokenizer tok;
LanguageModel model(&tok);

SamplerSystem sampler;

void CommandLoadModel(const std::vector<std::string>& args);
void CommandSaveModel(const std::vector<std::string>& args);
void CommandRead(const std::vector<std::string>& args);
void CommandTrim(const std::vector<std::string>& args);
void CommandClear(const std::vector<std::string>& args);

std::vector<int> context;
std::vector<std::vector<int>> focus;

int main() {
    srand(120);
    
    console.RegisterCommandFunction("read", &CommandRead);
    console.RegisterCommandFunction("load", &CommandLoadModel);
    console.RegisterCommandFunction("save", &CommandSaveModel);
    console.RegisterCommandFunction("trim", &CommandTrim);
    console.RegisterCommandFunction("clear", &CommandClear);
    
    const std::string modelFilename = "ds.model";
    const std::string attenFilename = "ds.attn";
    const std::string embedFilename = "ds.embed";
    
    if (FileExists(modelFilename) && FileExists(attenFilename) && FileExists(embedFilename)) {
        std::cout << "Loading model file... ";
        model.LoadFromFile(modelFilename);
        sampler.attention.LoadFromFile(attenFilename);
        sampler.embedding.LoadFromFile(embedFilename);
        std::cout << "complete\n\n";
    }
    
    while (true) {
        std::cout << "> ";
        std::string keyboard_string;
        
        if (!std::getline(std::cin, keyboard_string)) break;
        if (keyboard_string.empty() || keyboard_string[0] == ' ') 
            continue;
        
        std::vector<std::string> keyboardSplt = StringExplode(keyboard_string, ' ');
        if (keyboardSplt[0][0] == '/') {
            std::string command = keyboardSplt[0];
            keyboardSplt.erase(keyboardSplt.begin());
            command.erase(command.begin());
            if (!console.Run(command, keyboardSplt)) {
                std::cout << "Unknown command '"+command+"'\n\n";
            }
            continue;
        }
        
        // Add new context from the prompt
        Context contextConvert(&tok);
        contextConvert = keyboardSplt;
        std::vector<int> prompt = contextConvert.GetTokens();
        for (unsigned int i=0; i < prompt.size(); i++) 
            context.push_back( prompt[i] );
        
        // Broad pass - pull in chunks of relevant context
        //if (focus.size() < 1024) 
        //    model.GetContext(prompt, focus, 16);
        
        // Trim the focus
        //while (focus.size() > 8192) 
        //    focus.erase(focus.begin());
        
        if (context.size() == 0) {std::cout << "Context empty\n\n"; continue;}
        if (model.size() == 0)   {std::cout << "Model empty\n\n"; continue;}
        
        SamplerParameters params;
        params.temperatureHigh  = 0.7f;
        params.temperatureLow   = 0.1f;
        params.attentionRate    = 1.1f;
        params.embeddingRate    = 0.8f;
        
        const int sentenceMax   = 1;
        const int wordThreshold = 5;
        
        int sentenceCount = 0;
        int wordCount     = 0;
        
        bool isStreamLive  = true;
        bool firstRun      = true;
        bool doCapitalize  = true;
        int  lastToken     = -1;
        
        // Extract content tokens
        //std::vector<int> narrow;
        //model.GetRelevantContext(sampler.attention, prompt, narrow);
        
        // DEBUG - dump word stats from the prompt
        /*
        for (unsigned int i=0; i < prompt.size(); i++) {
            int token = prompt[i];
            std::string word = tok.tokenToWord[token];
            TokenInfo info = sampler.attention.GetTokenInfo(token);
            if (sampler.attention.IsContentLike(token, 0.0f)) {
                narrow.push_back(token);
                std::cout << "<" << word << ">   ";
            } else {
                std::cout << word << "   ";
            }
            
            std::cout << info.degree << "   " <<
                         info.relationScore << "   " <<
                         info.contentScore << "   " <<
                         info.pContent << "   " <<
                         info.pFunction << "   " <<
                         info.totalEdges << "\n";
            
        }
        std::cout << "\n";
        */
        
        // Narrow pass - pull in segments of relevant context using content tokens
        /*
        for (unsigned int i=0; i < narrow.size(); i++) {
            std::vector<int> token = {narrow[i]};
            model.GetContext(token, focus, 1);
            std::string word = tok.tokenToWord[token[0]];
        }
        */
        
        while (isStreamLive) {
            bool doEndSpace = false;
            
            //TokenDistribution dist = sampler.SampleNextTokenDistribution(context, focus, params, 5);
            //for (unsigned int i=0; i < dist.tokens.size(); i++) 
            //    std::cout << dist.weights[i] << "    " << tok.tokenToWord[dist.tokens[i]] << "\n";
            //break;
            
            int nextToken = sampler.SampleNextToken(context, model.mModel, params);
            
            // Handle special negative return codes first
            if (nextToken < 0) {
                if (nextToken == -3) std::cout << "Focus empty\n\n";
                if (nextToken == -2) std::cout << "Context empty\n\n";
                
                isStreamLive = false;
                continue;
            }
            
            // Guard against out-of-range positive indices
            if (nextToken >= tok.tokenToWord.size()) {
                std::cout << "Token index out of range: " << nextToken << "\n";
                break;
            }
            
            std::string word = tok.tokenToWord[nextToken];
            
            if (firstRun) {
                firstRun = false;
                if (word == "." || word == "?" || word == "!" || word == ",") {
                    context.push_back(nextToken);
                    continue;
                }
            }
            if (doCapitalize) {
                doCapitalize = false;
                StringCaseUpper(word);
            }
            
            context.push_back(nextToken);
            if (context.size() > 1024) 
                context.erase(context.begin());
            
            if (word == ",") 
                doEndSpace = true;
            
            if (word == "." || word == "!" || word == "?") {
                sentenceCount++;
                if (sentenceCount >= sentenceMax && wordCount >= wordThreshold) 
                    isStreamLive = false;
                doEndSpace = true;
                doCapitalize = true;
                
                if (lastToken == nextToken) 
                    continue;
            }
            lastToken = nextToken;
            
            if (!doEndSpace) 
                std::cout << " ";
            
            std::cout << word;
            
            wordCount++;
        }
        
        std::cout << "\n\n";
    }
    
    return 0;
}






//
// REPL functions

void CommandTrim(const std::vector<std::string>& args) {
    
    
    
}

void CommandClear(const std::vector<std::string>& args) {
    std::cout << "Context cleared.\n\n";
    context.clear();
    focus.clear();
}

void CommandLoadModel(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Usage: /load <filename>\n\n";
        return;
    }
    
    const std::string base = args[0];
    std::string modelFilename = base + ".model";
    std::string attenFilename = base + ".attn";
    std::string embedFilename = base + ".embed";
    
    std::cout << "Loading model '" << base << "'... ";
    model.LoadFromFile(modelFilename);
    sampler.attention.LoadFromFile(attenFilename);
    sampler.embedding.LoadFromFile(embedFilename);
    std::cout << "complete\n\n";
}

void CommandSaveModel(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Usage: /save <basename>\n\n";
        return;
    }
    
    const std::string base = args[0];
    std::string modelFilename = base + ".model";
    std::string attenFilename = base + ".attn";
    std::string embedFilename = base + ".embed";
    
    std::cout << "Saving model '" << base << "'... ";
    model.SaveToFile(modelFilename);
    sampler.attention.SaveToFile(attenFilename);
    sampler.embedding.SaveToFile(embedFilename);
    std::cout << "complete\n\n";
}

void CommandRead(const std::vector<std::string>& args) {
    const float strength = 2.4f;

    if (args.empty()) {
        std::cout << "Usage: /read <filename>\n\n";
        return;
    }
    
    std::string filename = args[0];
    if (!FileExists(filename)) {
        std::cout << "File not found: " << filename << "\n\n";
        return;
    }
    
    std::string source;
    FileTextLoad(filename, source);
    
    std::vector<std::string> corpus = StringExplode(source, ' ');
    tok.AddTokens(corpus);
    
    std::vector<std::vector<int>> encodings;
    encodings.reserve(corpus.size());
    
    for (unsigned int i=0; i < corpus.size(); i++) {
        std::vector<int> encoding;
        unsigned int counter=0;
        for (; i < corpus.size(); i++) {
            encoding.push_back( tok.wordToToken[corpus[i]] );
            counter++;
            if (counter >= 128 || 
                tok.wordToToken[corpus[i]] == tok.wordToToken["."] || 
                tok.wordToToken[corpus[i]] == tok.wordToToken["?"] || 
                tok.wordToToken[corpus[i]] == tok.wordToToken["!"]) 
                break;
        }
        encodings.push_back(encoding);
        
        //for (unsigned int a=0; a < encoding.size(); a++) 
        //    std::cout << " " << tok.tokenToWord[encoding[a]];
        //std::cout << "\n\n";
    }
    int counter=0;
    for (unsigned int e=0; e < encodings.size(); e++) {
        model.AddContext(encodings[e]);
        
        // Train attention and embeddings
        sampler.embedding.TrainOnSentence(encodings[e], encodings[e].size(), strength);
        sampler.attention.ProcessSequence(encodings[e]);
        
        counter++;
        if (counter > 128) {
            std::cout << e << " of " << encodings.size() << "\r";
            counter=0;
        }
    }
    std::cout << encodings.size() << " of " << encodings.size() << "\n\n";
    sampler.attention.NormalizeWeightsPerAnchor();
    
    sampler.attention.RenormalizeAll(0.9f);
}

