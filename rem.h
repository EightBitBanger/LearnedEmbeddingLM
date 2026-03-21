#ifndef _RELEARNABLEEMBEDDINGMATRIX_
#define _RELEARNABLEEMBEDDINGMATRIX_

#include <vector>

class RelearnableEmbeddingTransformer {
public:
    
    void Initiate(unsigned int layers, unsigned int width, unsigned int exp) {
        mLayers.clear();
        std::vector<float> il;
        std::vector<float> el;
        
        mLayerWidth = width;
        mExpansionWidth = width * exp;
        
        for (unsigned int i=0; i < width; i++) 
            il.push_back(0.0f);
        for (unsigned int e=0; e < mExpansionWidth; e++) 
            el.push_back(0.0f);
        
        for (unsigned int l=0; l < layers; l++) {
            mLayers.push_back(il);
            mLayers.push_back(el);
        }
        
        // Final output layer
        mLayers.push_back(il);
    }
    
    
private:
    
    unsigned int mLayerWidth;
    unsigned int mExpansionWidth;
    
    std::vector<std::vector<float>> mLayers;
};


#endif
