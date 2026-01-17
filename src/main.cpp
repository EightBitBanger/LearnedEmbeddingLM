#include <iostream>
#include <thread>

#include "platform.h"
#include "string.h"
#include "repl.h"

#include "sampler.h"
#include "tokenizer.h"
#include "languagemodel.h"
#include "gsa.h"

ReplCommandConsole console;

LanguageModel model;

SamplerSystem sampler(&model.attention, &model.embedding);

GlobalSemanticAlignment gsa(&model.tok, &sampler, &model.attention);

void CommandReadCorpus(const std::vector<std::string>& args);
void CommandLoadModel(const std::vector<std::string>& args);
void CommandSaveModel(const std::vector<std::string>& args);
void CommandClear(const std::vector<std::string>& args);
void CommandClip(const std::vector<std::string>& args);

std::vector<int> context;

std::vector<std::vector<int>> focus;

int main() {
    
    console.RegisterCommandFunction("read", &CommandReadCorpus);
    console.RegisterCommandFunction("load", &CommandLoadModel);
    console.RegisterCommandFunction("save", &CommandSaveModel);
    console.RegisterCommandFunction("clear", &CommandClear);
    console.RegisterCommandFunction("clip", &CommandClip);
    
    // Load the dataset model
    if (FileExists("ds.model")) {
        std::cout << "Loading model 'ds'\n\n";
        CommandLoadModel({"ds"});
    }
    
    // Sampler and paragraphing
    SamplerParameters params;
    params.temperatureHigh  = 1.1f;
    params.temperatureLow   = 0.7f;
    params.attentionRate    = 70.0f;
    params.embeddingRate    = 0.0f;
    
    const int sentenceMax    = 1;    // Number of sentences to generate
    const int wordThreshold  = 5;    // Sample at least as many tokens
    const int paragraphMax   = 3;    // Sentences per paragraph
    
    const int  wrapWidth     = 100;
    
    const std::string firstLineIndentStr = " ";
    const int indentLen = (int)firstLineIndentStr.size();
    
    
    
    while (true) {
        std::cout << "> ";
        std::string prompt;
        
        if (!std::getline(std::cin, prompt)) break;
        if (prompt.empty() || prompt[0] == ' ') 
            continue;
        
        // Explode by spaces
        std::vector<std::string> promptSplit = StringExplode(prompt, ' ');
        
        // Check prompt command
        if (promptSplit[0][0] == '/') {
            std::string command = promptSplit[0];
            promptSplit.erase(promptSplit.begin());
            command.erase(command.begin());
            if (!console.Run(command, promptSplit)) {
                std::cout << "Unknown function '" << command << "'\n\n";
            }
            continue;
        }
        
        // Output token stream for the whole prompt
        std::vector<int> promptTokens;
        promptTokens.reserve(promptSplit.size() * 2); // BPE expansion
        
        // Perform a byte-pair encoding pass on user input
        for (unsigned int i = 0; i < promptSplit.size(); i++) {
            const std::string& w = promptSplit[i];
            if (w.empty())
                continue;
            
            // Check for the whole word in the vocabulary
            if (model.tok.CheckWordExists(w)) {
                const int token = model.tok.GetToken(w);
                if (token >= 0) {
                    promptTokens.push_back(token);
                }
                continue;
            }
            
            // Decompose the unknown token into existing vocab tokens
            // Returns <unk> tokens on unknown parts
            std::vector<int> subTokens;
            const bool ok = model.tok.TokenizeWordBPE(w, subTokens, "<unk>", false, "##");
            
            if (!ok || subTokens.empty()) {
                std::cout << "!<" << w << ">\n";
                continue;
            }
            
            // Append subword tokens into the prompt stream
            for (unsigned int k = 0; k < subTokens.size(); k++) {
                promptTokens.push_back(subTokens[k]);
            }
        }
        
        // Check prompt question
        if (gsa.GetQuestionScore(promptTokens) > 0.5f) {
            std::vector<int> subjects = gsa.GetQuestionSubject(promptTokens, promptTokens.size());
            
            for (unsigned int i=0; i < subjects.size(); i++) 
                std::cout << " " << model.tok.tokenToWord[subjects[i]];
        }
        std::cout << "\n\n";
        
        // Add new context from the prompt
        for (unsigned int i=0; i < promptTokens.size(); i++) 
            context.push_back( promptTokens[i] );
        
        // Score the most relevant tokens from the prompt
        std::vector<std::pair<int, float>> scoredTokens;
        scoredTokens.reserve(promptTokens.size());
        
        for (unsigned int t = 0; t < promptTokens.size(); t++) {
            int token = promptTokens[t];
            float score = model.attention.GetContentScore(token);
            
            scoredTokens.emplace_back(token, score);
        }
        std::sort(scoredTokens.begin(), scoredTokens.end(),
            [](const std::pair<int, float>& a,
            const std::pair<int, float>& b) {
                return a.second > b.second;
            });
        
        std::vector<int> topics;
        topics.reserve(scoredTokens.size());
        float max = scoredTokens[scoredTokens.size()-1].second;
        float min = scoredTokens[0].second;
        float avg = (max + min) / 3.0f;
        if (avg >= max) avg = min;
        
        for (unsigned int t=0; t < scoredTokens.size(); t++) {
            const std::pair<int, float>& tPair = scoredTokens[t];
            int   token = tPair.first;
            float score = tPair.second;
            
            if (score < avg) 
                continue;
            
            // De-duplicate
            bool found=false;
            for (unsigned int d=0; d < topics.size(); d++) {
                if (topics[d] == token) {
                    found = true;
                    break;
                }
            }
            if (found == false) 
                topics.push_back(token);
        }
        
        // Testing agentic implementation
        /*
        std::string topicTok = "<topic>";
        std::string agent = "It looks like the user is asking about "+topicTok;
        
        // Inject topic
        if (topics.size() > 0) {
            unsigned int pos = agent.find(topicTok);
            agent.erase(pos, topicTok.size());
            std::string word = model.tok.tokenToWord[topics[0]];
            agent.insert(pos, word);
            
            std::cout << agent << "\n\n";
        }
        */
        
        // Gather some context relevant regions from the model
        std::vector<std::vector<int>> localFocalRegions = model.Extract(topics);
        for (unsigned int t=0; t < localFocalRegions.size(); t++) {
            focus.push_back(localFocalRegions[t]);
        }
        
        // Clip the focus
        while (focus.size() > (1024 * 2)) 
            focus.erase(focus.begin());
        
        int sentenceCount=0;
        int wordCount=0;
        int newlineCount=0;
        
        bool isStreamLive  = true;
        bool firstRun      = true;
        bool doCapitalize  = true;
        int  lastToken     = -1;
        
        // Formatting state
        bool atLineStart       = true;
        bool atParagraphStart  = true;
        int  lineLen           = 0;
        
        std::string lastWordPrinted;
        bool streamStarted = false;
        
        while (isStreamLive) {
            bool doEndSpace = false;
            
            int token = gsa.SampleAligned(context, focus, params);
            
            // Handle special negative return codes first
            if (token < 0) {
                if (token == -3) std::cout << "Focus empty\n\n";
                if (token == -2) std::cout << "Context empty\n\n";
                
                isStreamLive = false;
                continue;
            }
            
            // Check out-of-range
            if (token >= model.tok.tokenToWord.size()) {
                std::cout << "Token index out of range: " << token << "\n";
                break;
            }
            
            context.push_back(token);
            if (context.size() > 1024) 
                context.erase(context.begin());
            
            std::string word = model.tok.tokenToWord[token];
            
            // Skip leading punctuation until we hit a real word.
            // Tokens still remain in context (already pushed above).
            if (!streamStarted) {
                if (word.empty() || IsSkippableLeadingToken(word)) {
                    continue;
                }
                streamStarted = true;
            }
            
            if (doCapitalize) {
                if (!word.empty() && !IsNoSpaceBeforePunct(word) && !IsOpenBracket(word)) {
                    doCapitalize = false;
                    StringCaseUpper(word);
                }
            }
            
            bool needSpaceBefore = false;
            if (!atLineStart) {
                if (!IsNoSpaceBeforePunct(word)) {
                    // Don't force a space right after an opening bracket like "("
                    if (lastWordPrinted.empty() || !IsOpenBracket(lastWordPrinted)) {
                        needSpaceBefore = true;
                    }
                }
            }
            // If adding a space would overflow the line, wrap first
            int projected = lineLen;
            if (needSpaceBefore) projected += 1;
            projected += (int)word.size();
            
            if (!atLineStart && projected > wrapWidth) {
                std::cout << "\n";
                atLineStart = true;
                lineLen = 0;
            
                // New wrapped lines inside a paragraph are NOT indented
                atParagraphStart = false;
            
                // If the word would start a wrapped line, recompute indent rules
                needSpaceBefore = false;
            }
            
            // Apply indent ONLY when we are about to print the first token on the first line of a paragraph.
            if (atLineStart && atParagraphStart) {
                std::cout << firstLineIndentStr;
                lineLen += indentLen;
            }

            // Print optional space, then token
            if (needSpaceBefore) {
                std::cout << " ";
                lineLen += 1;
            }
            std::cout << word;
            lineLen += (int)word.size();
            
            atLineStart = false;
            lastWordPrinted = word;
            
            // Sentence end => maybe start a new paragraph + enable capitalization
            if (IsSentenceEnd(word)) {
                sentenceCount++;
                doCapitalize = true;
                
                if (sentenceCount >= sentenceMax && wordCount >= wordThreshold)
                    isStreamLive = false;
                
                // Paragraph break every N sentences (tune sentencesPerParagraph)
                if ((sentenceCount % paragraphMax) == 0) {
                    std::cout << "\n\n";
                    atLineStart = true;
                    atParagraphStart = true;
                    lineLen = 0;
                }
            }
            
            wordCount++;
            
        }
        
        std::cout << "\n\n";
        
    }
    
    return 0;
}

bool modelIsFinished=false;
bool AttentionFinished=false;
bool EmbeddingFinished=false;

void LoadModelThread(const std::string filename) {model.LoadFromFile(filename); modelIsFinished=true;}
void LoadAttentionThread(const std::string filename) {model.attention.LoadFromFile(filename); AttentionFinished=true;}
void LoadEmbeddingThread(const std::string filename) {model.embedding.LoadFromFile(filename); EmbeddingFinished=true;}

void CommandLoadModel(const std::vector<std::string>& args) {
    if (args.size() < 1) {
        std::cout << "Enter a model name to load\n";
        std::cout << " /load modelname\n\n";
        return;
    }
    const std::string modelFilename     = args[0] + ".model";
    const std::string attentionFilename = args[0] + ".attn";
    const std::string embeddingFilename = args[0] + ".embed";
    
    bool modelFile = FileExists(modelFilename);
    bool attnFile  = FileExists(attentionFilename);
    bool embedFile = FileExists(embeddingFilename);
    
    if (!modelFile || !attnFile || !embedFile) {
        std::cout << "Model file not found '";
        if (!modelFile) std::cout << modelFilename << "'\n\n";
        if (!attnFile)  std::cout << attentionFilename << "'\n";
        if (!embedFile) std::cout << embeddingFilename << "'\n";
        return;
    }
    
    modelIsFinished   = false;
    AttentionFinished = false;
    EmbeddingFinished = false;
    
    std::thread threadLoadModel(LoadModelThread,     modelFilename);
    std::thread threadLoadAtten(LoadAttentionThread, attentionFilename);
    std::thread threadLoadEmbed(LoadEmbeddingThread, embeddingFilename);
    
    threadLoadModel.join();
    threadLoadAtten.join();
    threadLoadEmbed.join();
    
    gsa = GlobalSemanticAlignment(&model.tok, &sampler, &model.attention);
}

void CommandSaveModel(const std::vector<std::string>& args) {
    if (args.size() < 1) {
        std::cout << "Enter a model name to save\n";
        std::cout << " /save modelname\n\n";
        return;
    }
    const std::string modelFilename     = args[0] + ".model";
    const std::string attentionFilename = args[0] + ".attn";
    const std::string embeddingFilename = args[0] + ".embed";
    
    model.attention.SaveToFile(attentionFilename);
    model.embedding.SaveToFile(embeddingFilename);
    model.SaveToFile(modelFilename);
}


void CommandReadCorpus(const std::vector<std::string>& args) {
    if (args.size() < 1) {
        std::cout << "Enter a filename to read\n";
        return;
    }
    
    if (!FileExists(args[0])) {
        std::cout << "File not found '" << args[0] << "'\n\n";
        return;
    }
    
    const unsigned int maxSpanWidth = 128;
    
    std::string rawText;
    FileTextLoad(args[0], rawText);
    std::vector<std::string> corpus = StringExplode(rawText, ' ');
    model.tok.AddTokens(corpus);
    
    std::vector<std::vector<int>> encodings;
    encodings.reserve(corpus.size());
    
    for (unsigned int i=0; i < corpus.size(); i++) {
        std::vector<int> encoding;
        unsigned int counter=0;
        for (; i < corpus.size(); i++) {
            const std::string word = corpus[i];
            
            encoding.push_back( model.tok.wordToToken[word] );
            counter++;
            if (counter >= maxSpanWidth || 
                model.tok.wordToToken[word] == model.tok.wordToToken["."] || 
                model.tok.wordToToken[word] == model.tok.wordToToken["?"] || 
                model.tok.wordToToken[word] == model.tok.wordToToken["!"]) 
                break;
        }
        encodings.push_back(encoding);
    }
    
    float embeddingStrength = 0.8f;
    
    // Process encodings
    unsigned int counter=0;
    for (unsigned int e=0; e < encodings.size(); e++) {
        std::vector<int>& encoding = encodings[e];
        
        model.ProcessSequence( encoding, 1.0f );
        model.attention.ProcessSequence( encoding );
        model.embedding.TrainOnSentence(encodings[e], encodings[e].size(), embeddingStrength);
        
        counter++;
        if (counter > 2048) {
            counter=0;
            std::cout << e << " of " << encodings.size() << "\r";
        }
    }
    std::cout << encodings.size() << " of " << encodings.size() << "\n\n";
    
    model.attention.RecomputeRoleScores();
}



void CommandClear(const std::vector<std::string>& args) {
    context.clear();
    focus.clear();
    
    std::cout << " Cleared the context." << "\n\n";
}

void CommandClip(const std::vector<std::string>& args) {
    unsigned int numberOfTokensRemoved = model.attention.PruneLowInteractionTokens(3, 0.001f, false);
    std::cout << "tokens culled  " << numberOfTokensRemoved << "\n\n";
}

