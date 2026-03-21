#include "attention.h"
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_set>

void AttentionSystem::ProcessSequence(const std::vector<int>& tokens) {
    const int N = (int)tokens.size();
    if (N <= 1) 
        return;
    
    // Local window radius around the anchor token.
    const int windowRadius = (int)tokens.size() - 1;
    
    updateStep++;
    
    for (unsigned int h = 0; h < (unsigned int)windowRadius; h++) {
        
        // Pick a random anchor index
        int i = std::rand() % N;
        int anchor = tokens[(unsigned int)i];
        
        // Pick a random neighbor index in a window around i
        int j = i;
        for (int a = 0; a < windowRadius && j == i; a++) {
            int off = (std::rand() % (2 * windowRadius + 1)) - windowRadius; // [-R, R]
            if (off == 0) {
                continue;
            }
            
            int candidate = i + off;
            if (candidate < 0 || candidate >= N) {
                continue;
            }
            
            j = candidate;
        }
        
        if (j == i) {
            continue; // failed to find a different index
        }
        
        int neighbor = tokens[(unsigned int)j];
        int offset   = j - i; // signed distance
        
        float d = std::fabs((float)offset);
        float weight = baseWeight / (1.0f + d * falloff);
        
        AttentionKey key{anchor, neighbor, offset};
        
        AttentionEdge &edge = attention[key];
        edge.weight        += weight;
        edge.count         += 1u;
        edge.lastUpdateStep = updateStep;
        
        // Update simple per-token role stats.
        TokenRoleStats &sa = tokenStats[anchor];
        sa.asAnchorCount += 1u;
        sa.totalEdges    += 1u;
        
        TokenRoleStats &sn = tokenStats[neighbor];
        sn.asNeighborCount += 1u;
        sn.totalEdges      += 1u;
    }
}

static unsigned int HalfCeil(unsigned int v) {
    return (v + 1u) >> 1; // avoids getting stuck at 0 for small counts
}

void AttentionSystem::RenormalizeAll(float weightScale) {
    // Scale edge data
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::iterator it = attention.begin();
         it != attention.end(); ++it) {
        it->second.count  = HalfCeil(it->second.count);
        it->second.weight *= weightScale;
    }
    
    // Scale token stats
    for (std::unordered_map<int, TokenRoleStats>::iterator it = tokenStats.begin();
         it != tokenStats.end(); ++it) {
        TokenRoleStats& st = it->second;
        st.asAnchorCount   = HalfCeil(st.asAnchorCount);
        st.asNeighborCount = HalfCeil(st.asNeighborCount);
        st.totalEdges      = HalfCeil(st.totalEdges);
    }
}

void AttentionSystem::Clear() {
    attention.clear();
    tokenStats.clear();
    updateStep = 0u;
}

// Return weight for a specific (anchor, candidate, offset) triple.
float AttentionSystem::GetScore(int anchor, int candidate, int offset) const {
    AttentionKey key{anchor, candidate, offset};
    std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
        attention.find(key);
    if (it == attention.end()) {
        return 0.0f;
    }
    return it->second.weight;
}

// Aggregate score over all offsets for (anchor, candidate).
float AttentionSystem::GetScore(int anchor, int candidate) const {
    float total = 0.0f;
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
             attention.begin();
         it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        if (k.anchor == anchor && k.neighbor == candidate) {
            total += it->second.weight;
        }
    }
    return total;
}

float AttentionSystem::GetScore(const std::vector<int>& context, int token_j) const {
    if (context.empty()) {
        return 0.0f;
    }
    
    const float decay = 0.7f;
    
    float totalScore  = 0.0f;
    float totalWeight = 0.0f;
    
    float w = 1.0f;
    const int nextIndex = (int)context.size(); // index where token_j would appear
    
    // Walk backwards through context
    for (int i = (int)context.size() - 1; i >= 0; --i) {
        int anchor = context[(unsigned int)i];
        
        // Compute the offset the candidate would have relative to this anchor.
        int offset = nextIndex - i; // j - i, where j == nextIndex
        
        float s = GetScore(anchor, token_j, offset);
        if (s != 0.0f) {
            totalScore  += s * w;
            totalWeight += w;
        }
        
        w *= decay;
    }
    
    if (totalWeight <= 0.0f) {
        return 0.0f;
    }
    return totalScore / totalWeight;
}

int AttentionSystem::GetNextToken(const std::vector<int>& context,
                                  const std::vector<int>& allTokens) {
    int   bestToken = -1;
    float bestScore = -1e30f;
    
    for (std::size_t i = 0; i < allTokens.size(); ++i) {
        int t = allTokens[i];
        float s = GetScore(context, t);
        if (s > bestScore) {
            bestScore = s;
            bestToken = t;
        }
    }
    
    return bestToken;
}

void AttentionSystem::NormalizeWeightsPerAnchor() {
    // First, accumulate total weight per anchor.
    std::unordered_map<int, float> sumPerAnchor;
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
             attention.begin();
         it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        sumPerAnchor[k.anchor] += it->second.weight;
    }
    
    // Then scale each edge by the sum for its anchor.
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::iterator it =
             attention.begin();
         it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        float sum = sumPerAnchor[k.anchor];
        if (sum > 0.0f) {
            it->second.weight *= (1.0f / sum);
        }
    }
}

// Set a specific (tokenA, tokenB, offset) score.
void AttentionSystem::SetScore(int tokenA, int tokenB, int offset, float score) {
    AttentionKey key{tokenA, tokenB, offset};
    AttentionEdge &edge = attention[key];
    edge.weight = score;
    // keep count / lastUpdateStep as-is or reset if you want:
    // edge.count = 0;
    // edge.lastUpdateStep = updateStep;
}

// Set aggregate score for (tokenA, tokenB) by distributing across existing offsets.
void AttentionSystem::SetScore(int tokenA, int tokenB, float score) {
    // Count how many offsets exist for (tokenA, tokenB).
    float count = 0.0f;
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
             attention.begin();
         it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        if (k.anchor == tokenA && k.neighbor == tokenB) {
            count += 1.0f;
        }
    }
    if (count <= 0.0f) {
        return;
    }
    
    float per = score / count;
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::iterator it =
             attention.begin();
         it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        if (k.anchor == tokenA && k.neighbor == tokenB) {
            it->second.weight = per;
        }
    }
}

// Scale a specific (tokenA, tokenB, offset) association.
void AttentionSystem::AdjustScore(int tokenA, int tokenB, int offset, float multiplier) {
    AttentionKey key{tokenA, tokenB, offset};
    std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::iterator it =
        attention.find(key);
    if (it == attention.end()) {
        return;
    }
    
    it->second.weight *= multiplier;
}

// Scale all offsets for (tokenA, tokenB).
void AttentionSystem::AdjustScore(int tokenA, int tokenB, float multiplier) {
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::iterator it =
             attention.begin();
         it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        if (k.anchor == tokenA && k.neighbor == tokenB) {
            it->second.weight *= multiplier;
        }
    }
}

float AttentionSystem::GetAverageOffset(int tokenA, int tokenB) const {
    float sumW  = 0.0f;
    float sumWO = 0.0f;
    
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
            attention.begin();
        it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        if (k.anchor == tokenA && k.neighbor == tokenB) {
            float w = it->second.weight;
            sumW  += w;
            sumWO += w * (float)k.offset;
        }
    }
    
    if (sumW <= 0.0f) {
        return 0.0f;
    }
    return sumWO / sumW;
}

void AttentionSystem::RecomputeRoleScores(void) {
    // Approximate degree for each token by counting unique neighbors/anchors.
    std::unordered_map<int, std::unordered_set<int> > neighborsOf;
    std::unordered_map<int, std::unordered_set<int> > anchorsOf;
    
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
            attention.begin();
        it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        neighborsOf[k.anchor].insert(k.neighbor);
        anchorsOf[k.neighbor].insert(k.anchor);
    }
    
    std::unordered_map<int, unsigned int> degree;
    
    // Anchor degrees: number of distinct neighbors.
    for (std::unordered_map<int, std::unordered_set<int> >::const_iterator it =
            neighborsOf.begin();
        it != neighborsOf.end(); ++it) {
        int anchor = it->first;
        degree[anchor] += (unsigned int)it->second.size();
    }
    
    // Neighbor degrees: number of distinct anchors they are connected to.
    for (std::unordered_map<int, std::unordered_set<int> >::const_iterator it =
            anchorsOf.begin();
        it != anchorsOf.end(); ++it) {
        int neighbor = it->first;
        degree[neighbor] += (unsigned int)it->second.size();
    }
    
    // Combine with tokenStats to derive relation/content-ish scores.
    std::unordered_map<int, TokenRoleStats>::iterator itT =
        tokenStats.begin();
    for (; itT != tokenStats.end(); ++itT) {
        int token = itT->first;
        TokenRoleStats &st = itT->second;
        
        unsigned int deg = 0u;
        std::unordered_map<int, unsigned int>::const_iterator itD =
            degree.find(token);
        if (itD != degree.end()) {
            deg = itD->second;
        }
        
        st.degree = (float)deg;
        
        float anchorF   = (float)st.asAnchorCount;
        float neighborF = (float)st.asNeighborCount;
        float totalF    = anchorF + neighborF + 1.0f;
        
        // Very simple heuristic:
        //  - Higher degree and balanced anchor/neighbor usage
        //    -> more "relation/glue-like".
        st.relationScore = st.degree / totalF;
        
        // Content-ish tokens roughly the inverse.
        st.contentScore = 1.0f / (1.0f + st.relationScore);
    }
}

const TokenRoleStats* AttentionSystem::GetTokenStats(int token) const {
    std::unordered_map<int, TokenRoleStats>::const_iterator it =
        tokenStats.find(token);
    if (it == tokenStats.end()) {
        return NULL;
    }
    return &(it->second);
}

TokenInfo AttentionSystem::GetTokenInfo(int token) const {
    TokenInfo info;
    
    const TokenRoleStats* stats = GetTokenStats(token);
    if (stats == NULL) {
        // info.hasStats is already false, probs at 0.5 / 0.5
        return info;
    }
    
    info.hasStats      = true;
    info.degree        = stats->degree;
    info.relationScore = stats->relationScore;
    info.contentScore  = stats->contentScore;
    info.totalEdges    = stats->totalEdges;
    
    // Clamp contentScore into [0,1] just to be safe.
    float content = stats->contentScore;
    if (content < 0.0f) content = 0.0f;
    if (content > 1.0f) content = 1.0f;
    
    // Interpret contentScore directly as "content-ness".
    // Then "function-ness" is just the complement.
    float baseContent  = content;           // [0,1]
    float baseFunction = 1.0f - baseContent; // [0,1]
    
    // Compute a confidence factor in [0,1] based on how many edges we have.
    float e = (float)stats->totalEdges;
    float conf = e / (e + 1); // saturating to <1; more edges -> higher conf
    if (conf < 0.0f) conf = 0.0f;
    if (conf > 1.0f) conf = 1.0f;
    
    info.confidence = conf;
    
    // Blend the base scores toward 0.5/0.5 as confidence drops:
    //
    //   final = 0.5 + (base - 0.5) * confidence
    //
    // So:
    //   - if confidence = 0  ->  final = 0.5 (unknown)
    //   - if confidence = 1  ->  final = base (fully trust scores)
    float contentFinal  = 0.5f + (baseContent  - 0.5f) * conf;
    float functionFinal = 0.5f + (baseFunction - 0.5f) * conf;
    
    // Optional tiny renormalization, just in case of rounding error.
    float sum = contentFinal + functionFinal;
    if (sum > 0.0f) {
        contentFinal  /= sum;
        functionFinal /= sum;
    }
    
    info.pContent  = contentFinal;
    info.pFunction = functionFinal;
    
    return info;
}

bool AttentionSystem::IsContentLike(int token, float threshold) const {
    TokenInfo info = GetTokenInfo(token);
    if (info.contentScore   < 0.85f + threshold && 
        info.degree         < 10000 && 
        info.relationScore  > 0.1f) 
        return true;
    
    return false;
}

bool AttentionSystem::SaveToFile(const std::string& filename) const {
    FILE* f = std::fopen(filename.c_str(), "wb");
    if (!f) {
        return false;
    }

    // Basic parameters
    uint32_t np    = (uint32_t)n_points;
    uint32_t step  = (uint32_t)updateStep;

    std::fwrite(&np,         sizeof(uint32_t), 1, f);
    std::fwrite(&baseWeight, sizeof(float),    1, f);
    std::fwrite(&falloff,    sizeof(float),    1, f);
    std::fwrite(&step,       sizeof(uint32_t), 1, f);

    // We keep the same on-disk structure as before:
    // [nAnchors]
    //   anchor, [nNeighbors]
    //     neighbor, [nOffsets]
    //       offset, edge
    //
    // So first group edges by (anchor, neighbor).
    std::unordered_map<int,
        std::unordered_map<int, std::vector<std::pair<int, AttentionEdge> > > > grouped;

    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
             attention.begin();
         it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        const AttentionEdge &e = it->second;
        grouped[k.anchor][k.neighbor].push_back(std::make_pair(k.offset, e));
    }

    uint32_t nAnchors = (uint32_t)grouped.size();
    std::fwrite(&nAnchors, sizeof(uint32_t), 1, f);

    for (const auto& anchorPair : grouped) {
        int anchor = anchorPair.first;
        const auto& neighbors = anchorPair.second;

        std::fwrite(&anchor, sizeof(int), 1, f);

        uint32_t nNeighbors = (uint32_t)neighbors.size();
        std::fwrite(&nNeighbors, sizeof(uint32_t), 1, f);

        for (const auto& neighborPair : neighbors) {
            int neighbor = neighborPair.first;
            const std::vector<std::pair<int, AttentionEdge> > &offsets =
                neighborPair.second;

            std::fwrite(&neighbor, sizeof(int), 1, f);

            uint32_t nOffsets = (uint32_t)offsets.size();
            std::fwrite(&nOffsets, sizeof(uint32_t), 1, f);

            for (std::size_t i = 0; i < offsets.size(); ++i) {
                int offset = offsets[i].first;
                const AttentionEdge &edge = offsets[i].second;

                std::fwrite(&offset,              sizeof(int),      1, f);
                std::fwrite(&edge.weight,         sizeof(float),    1, f);
                std::fwrite(&edge.count,          sizeof(uint32_t), 1, f);
                std::fwrite(&edge.lastUpdateStep, sizeof(uint32_t), 1, f);
            }
        }
    }

    std::fclose(f);
    return true;
}

bool AttentionSystem::LoadFromFile(const std::string& filename) {
    FILE* f = std::fopen(filename.c_str(), "rb");
    if (!f) {
        return false;
    }

    Clear(); // clear existing graph + stats

    uint32_t np    = 0;
    uint32_t step  = 0;

    if (std::fread(&np,         sizeof(uint32_t), 1, f) != 1 ||
        std::fread(&baseWeight, sizeof(float),    1, f) != 1 ||
        std::fread(&falloff,    sizeof(float),    1, f) != 1 ||
        std::fread(&step,       sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    n_points   = (unsigned int)np;
    updateStep = (unsigned int)step;

    uint32_t nAnchors = 0;
    if (std::fread(&nAnchors, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    for (uint32_t a = 0; a < nAnchors; ++a) {
        int anchor = 0;
        if (std::fread(&anchor, sizeof(int), 1, f) != 1) {
            std::fclose(f);
            return false;
        }

        uint32_t nNeighbors = 0;
        if (std::fread(&nNeighbors, sizeof(uint32_t), 1, f) != 1) {
            std::fclose(f);
            return false;
        }

        for (uint32_t n = 0; n < nNeighbors; ++n) {
            int neighbor = 0;
            if (std::fread(&neighbor, sizeof(int), 1, f) != 1) {
                std::fclose(f);
                return false;
            }

            uint32_t nOffsets = 0;
            if (std::fread(&nOffsets, sizeof(uint32_t), 1, f) != 1) {
                std::fclose(f);
                return false;
            }

            for (uint32_t o = 0; o < nOffsets; ++o) {
                int offset = 0;
                AttentionEdge edge;

                if (std::fread(&offset,              sizeof(int),      1, f) != 1 ||
                    std::fread(&edge.weight,         sizeof(float),    1, f) != 1 ||
                    std::fread(&edge.count,          sizeof(uint32_t), 1, f) != 1 ||
                    std::fread(&edge.lastUpdateStep, sizeof(uint32_t), 1, f) != 1) {
                    std::fclose(f);
                    return false;
                }

                AttentionKey key{anchor, neighbor, offset};
                attention[key] = edge;
            }
        }
    }

    std::fclose(f);

    // Rebuild tokenStats from the edges we just loaded.
    tokenStats.clear();
    for (std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash>::const_iterator it =
            attention.begin();
        it != attention.end(); ++it) {
        const AttentionKey &k = it->first;
        const AttentionEdge &edge = it->second;
        
        unsigned int c = edge.count;
        if (c == 0u) {
            c = 1u; // if count was never tracked, at least register one
        }
        
        TokenRoleStats& sa = tokenStats[k.anchor];
        sa.asAnchorCount += c;
        sa.totalEdges    += c;
        
        TokenRoleStats& sn = tokenStats[k.neighbor];
        sn.asNeighborCount += c;
        sn.totalEdges      += c;
    }
    
    // Recompute degree / relationScore / contentScore from the loaded graph.
    RecomputeRoleScores();
    
    return true;
}
