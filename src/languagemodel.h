#ifndef _LANGUAGE_MODEL__
#define _LANGUAGE_MODEL__

#include <vector>
#include <string>

#include "tokenizer.h"
#include "attention.h"

class LanguageModel {
public:
    
    // Build a "focus" list of tokens that are associated with the given context.
    // Returns true if at least one matching clip was found.
    bool GetContext(const std::vector<int>& context, std::vector<std::vector<int>>& focus, unsigned int range);
    
    // Branch off into other relevant contexts
    bool GetRelevantContext(AttentionSystem& attention, const std::vector<int>& context, std::vector<int>& focus);
    
    // Add a context span to the model.
    void AddContext(const std::vector<int>& context);
    
    // Save the model data to a file.
    bool SaveToFile(const std::string& filename) const;
    
    // Load the model data from a file.
    bool LoadFromFile(const std::string& filename);
    
    LanguageModel(Tokenizer* tokenizer);
    
    // Get the size of the model
    unsigned int size(void) const;
    
    std::vector<std::vector<int>> mModel;
    
private:
    friend class SamplerSystem;
    Tokenizer* tok;
    
};

#endif
