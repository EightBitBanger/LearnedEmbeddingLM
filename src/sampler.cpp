#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "sampler.h"
#include "gsa.h"

SamplerSystem::SamplerSystem(AttentionSystem* attentionPtr, EmbeddingSystem* embeddingPtr) : 
    attention(attentionPtr),
    embedding(embeddingPtr){}

int SamplerSystem::GetSentenceStart(int contextSize, int maxSentenceLen) const {
    int sentenceStart = 0;
    if (contextSize > maxSentenceLen) {
        sentenceStart = contextSize - maxSentenceLen;
    }
    return sentenceStart;
}

static bool TokenInList(int t, const std::vector<int>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] == t) return true;
    }
    return false;
}

void SamplerSystem::ComputeSpanBestMatches(
    const std::vector<int>& context,
    int sentenceStart,
    int maxSentenceLen,
    const std::vector<std::vector<int>>& focus,
    std::vector<int>& spanBestLen,
    int& globalBestLen,
    int& globalBestSpan) const
{
    const int contextSize = static_cast<int>(context.size());
    spanBestLen.assign(focus.size(), 0);
    globalBestLen  = 0;
    globalBestSpan = -1;

    for (std::size_t s = 0; s < focus.size(); ++s) {
        const std::vector<int>& span = focus[s];
        const int spanSize = static_cast<int>(span.size());
        if (spanSize < 2) {
            continue; // must have at least "token + nextToken"
        }

        int bestForThisSpan = 0;

        for (int i = 0; i < spanSize - 1; ++i) {
            int spanIdx  = i;
            int ctxIdx   = contextSize - 1;
            int matchLen = 0;

            while (spanIdx >= 0 &&
                   ctxIdx >= sentenceStart &&
                   matchLen < maxSentenceLen &&
                   span[static_cast<std::size_t>(spanIdx)] ==
                   context[static_cast<std::size_t>(ctxIdx)]) {
                ++matchLen;
                --spanIdx;
                --ctxIdx;
            }

            if (matchLen > bestForThisSpan) {
                bestForThisSpan = matchLen;
            }
        }

        spanBestLen[s] = bestForThisSpan;

        if (bestForThisSpan > globalBestLen) {
            globalBestLen  = bestForThisSpan;
            globalBestSpan = static_cast<int>(s);
        }
    }
}

void SamplerSystem::BuildScoreMaps(
    const std::vector<int>& context,
    int sentenceStart,
    int maxSentenceLen,
    const std::vector<std::vector<int>>& focus,
    const std::vector<int>& spanBestLen,
    int globalBestLen,
    int globalBestSpan,
    std::unordered_map<int, double>& lockedScores,
    std::unordered_map<int, double>& allScores) const
{
    lockedScores.clear();
    allScores.clear();

    if (globalBestLen <= 0 || globalBestSpan < 0) {
        return;
    }

    const int contextSize = static_cast<int>(context.size());

    for (std::size_t s = 0; s < focus.size(); ++s) {
        const std::vector<int>& span = focus[s];
        const int spanSize = static_cast<int>(span.size());
        if (spanSize < 2) {
            continue;
        }

        for (int i = 0; i < spanSize - 1; ++i) {
            int spanIdx  = i;
            int ctxIdx   = contextSize - 1;
            int matchLen = 0;

            while (spanIdx >= 0 &&
                   ctxIdx >= sentenceStart &&
                   matchLen < maxSentenceLen &&
                   span[static_cast<std::size_t>(spanIdx)] ==
                   context[static_cast<std::size_t>(ctxIdx)]) {
                ++matchLen;
                --spanIdx;
                --ctxIdx;
            }

            if (matchLen <= 0) {
                continue;
            }

            int nextToken = span[i + 1];

            // Quadratically emphasize longer matches:
            double weight = 1.0 + static_cast<double>(matchLen) *
                                     static_cast<double>(matchLen);

            // All matches contribute to the "looser" pool:
            allScores[nextToken] += weight;

            // Only the best span at the global best match length
            // contributes to the "locked" pool:
            if (static_cast<int>(s) == globalBestSpan &&
                matchLen == globalBestLen) {
                lockedScores[nextToken] += weight;
            }
        }
    }
}

void SamplerSystem::FallbackToFrequencyScores(
    const std::vector<std::vector<int>>& focus,
    std::unordered_map<int, double>& allScores,
    int& globalBestLen) const
{
    if (!allScores.empty()) {
        return;
    }

    std::unordered_map<int, int> freq;

    for (std::size_t s = 0; s < focus.size(); ++s) {
        const std::vector<int>& span = focus[s];
        for (std::size_t i = 0; i < span.size(); ++i) {
            ++freq[span[i]];
        }
    }

    if (freq.empty()) {
        return;
    }

    allScores.clear();
    for (std::unordered_map<int, int>::const_iterator it = freq.begin();
         it != freq.end(); ++it) {
        allScores[it->first] = static_cast<double>(it->second);
    }

    // In this fallback case, treat as very weak match.
    globalBestLen = 0;
}

void SamplerSystem::ChooseScoreSource(
    int globalBestLen,
    const std::unordered_map<int, double>& lockedScores,
    const std::unordered_map<int, double>& allScores,
    const SamplerParameters& params,
    bool& useLockedScores,
    float& effectiveTemp) const
{
    const int lockThreshold = 3; // tune this for stricter/looser locking

    useLockedScores = false;
    effectiveTemp   = params.temperatureHigh;

    if (globalBestLen >= lockThreshold && !lockedScores.empty()) {
        // Strong lock: only use the best span's continuations, at low temp.
        useLockedScores = true;
        effectiveTemp   = params.temperatureLow;
    } else if (globalBestLen == 2) {
        // Medium confidence: use all scores,
        // slightly cooler than full random.
        effectiveTemp = (params.temperatureLow + params.temperatureHigh) * 0.5f;
    } else {
        // globalBestLen == 1 or 0 -> weak or no lock: fully loose search.
        effectiveTemp = params.temperatureHigh;
    }

    if (effectiveTemp < 1e-3f) {
        effectiveTemp = 1e-3f;
    }
}




void SamplerSystem::BuildTokenDistribution(
    const std::vector<int>& context,
    const std::unordered_map<int, double>& baseScores,
    const SamplerParameters& params,
    float effectiveTemp,
    std::vector<int>& tokens,
    std::vector<double>& weights,
    double& totalWeight) const
{
    tokens.clear();
    weights.clear();
    totalWeight = 0.0;

    if (baseScores.empty()) {
        return;
    }

    // Clamp temperature to a small positive value so 1/temperature is stable.
    if (effectiveTemp < 1e-3f) {
        effectiveTemp = 1e-3f;
    }

    // -------------------------------------------------------------------------
    // 1) Find max base score (for normalization).
    // -------------------------------------------------------------------------
    double maxBase = 0.0;
    std::unordered_map<int, double>::const_iterator itBase;
    for (itBase = baseScores.begin(); itBase != baseScores.end(); ++itBase) {
        double baseVal = itBase->second;
        if (baseVal > maxBase) {
            maxBase = baseVal;
        }
    }

    // If we somehow have no positive base scores, fall back to uniform.
    if (maxBase <= 0.0) {
        for (std::unordered_map<int, double>::const_iterator it = baseScores.begin();
             it != baseScores.end(); ++it) {
            tokens.push_back(it->first);
            weights.push_back(1.0);
            totalWeight += 1.0;
        }
        return;
    }

    // -------------------------------------------------------------------------
    // 2) Compute mixing weights for base / attention / embedding.
    //
    // We treat base as weight 1.0 and scale att/emb by their user params,
    // then normalize so (wBase + wAtt + wEmb) = 1.
    // -------------------------------------------------------------------------
    double wBase = 1.0;
    double wAtt  = 0.0;
    double wEmb  = 0.0;

    if (params.attentionRate > 0.0f) {
        wAtt = static_cast<double>(params.attentionRate);
        if (wAtt < 0.0) {
            wAtt = 0.0;
        }
    }

    if (params.embeddingRate > 0.0f) {
        wEmb = static_cast<double>(params.embeddingRate);
        if (wEmb < 0.0) {
            wEmb = 0.0;
        }
    }

    double wSum = wBase + wAtt + wEmb;
    if (wSum <= 0.0) {
        wSum = 1.0;
    }
    wBase /= wSum;
    wAtt  /= wSum;
    wEmb  /= wSum;

    // -------------------------------------------------------------------------
    // 3) Precompute max attention score for normalization (if attention is used).
    // -------------------------------------------------------------------------
    float maxAttRaw = 0.0f;
    if (wAtt > 0.0) {
        std::unordered_map<int, double>::const_iterator it;
        for (it = baseScores.begin(); it != baseScores.end(); ++it) {
            int token = it->first;
            float attScore = attention->GetScore(context, token);
            if (attScore > maxAttRaw) {
                maxAttRaw = attScore;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4) Build a context embedding (average of context token embeddings).
    // -------------------------------------------------------------------------
    bool haveContextEmbedding = false;
    Embedding contextEmbedding;

    if (wEmb > 0.0 && embedding->size() > 0 && !context.empty()) {
        // Zero-init
        for (int d = 0; d < EMBEDDING_WIDTH; ++d) {
            contextEmbedding.v[d] = 0.0f;
        }

        int usedCount = 0;
        std::size_t i;
        for (i = 0; i < context.size(); ++i) {
            int ctxToken = context[i];
            const Embedding* ctxEmb = embedding->GetEmbeddingPtr(ctxToken);
            if (ctxEmb == NULL) {
                continue;
            }
            ++usedCount;
            for (int d = 0; d < EMBEDDING_WIDTH; ++d) {
                contextEmbedding.v[d] += ctxEmb->v[d];
            }
        }

        if (usedCount > 0) {
            float invCount = 1.0f / static_cast<float>(usedCount);
            for (int d = 0; d < EMBEDDING_WIDTH; ++d) {
                contextEmbedding.v[d] *= invCount;
            }

            // Normalize context embedding to unit length.
            float norm = 0.0f;
            for (int d = 0; d < EMBEDDING_WIDTH; ++d) {
                float val = contextEmbedding.v[d];
                norm += val * val;
            }

            if (norm > 0.0f) {
                norm = std::sqrt(norm);
                float invNorm = 1.0f / norm;
                for (int d = 0; d < EMBEDDING_WIDTH; ++d) {
                    contextEmbedding.v[d] *= invNorm;
                }
                haveContextEmbedding = true;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5) Precompute embedding similarities range (min / max) for normalization.
    //    We use cosine similarity between contextEmbedding and token embedding.
    // -------------------------------------------------------------------------
    double minEmbRaw = 0.0;
    double maxEmbRaw = 0.0;
    bool haveEmbRange = false;

    if (wEmb > 0.0 && haveContextEmbedding) {
        std::unordered_map<int, double>::const_iterator it;
        for (it = baseScores.begin(); it != baseScores.end(); ++it) {
            int token = it->first;
            const Embedding* tokenEmb = embedding->GetEmbeddingPtr(token);
            if (tokenEmb == NULL) {
                continue;
            }

            float dot = 0.0f;
            float normToken = 0.0f;
            int d;
            for (d = 0; d < EMBEDDING_WIDTH; ++d) {
                float tv = tokenEmb->v[d];
                dot       += contextEmbedding.v[d] * tv;
                normToken += tv * tv;
            }

            if (normToken <= 0.0f) {
                continue;
            }

            float tokenNorm = std::sqrt(normToken);
            if (tokenNorm <= 0.0f) {
                continue;
            }

            float sim = dot / tokenNorm; // contextEmbedding is already unit length

            if (!haveEmbRange) {
                minEmbRaw = sim;
                maxEmbRaw = sim;
                haveEmbRange = true;
            } else {
                if (sim < minEmbRaw) {
                    minEmbRaw = sim;
                }
                if (sim > maxEmbRaw) {
                    maxEmbRaw = sim;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 6) Build final distribution over tokens.
    // -------------------------------------------------------------------------
    std::unordered_map<int, double>::const_iterator it;
    for (it = baseScores.begin(); it != baseScores.end(); ++it) {
        int token = it->first;
        double baseVal = it->second;

        if (baseVal <= 0.0) {
            continue;
        }

        // Base normalized to [0, 1].
        double baseNorm = baseVal / maxBase;
        if (baseNorm < 0.0) {
            baseNorm = 0.0;
        }
        if (baseNorm > 1.0) {
            baseNorm = 1.0;
        }

        // Attention normalized to [0, 1].
        double attNorm = 0.0;
        if (wAtt > 0.0 && maxAttRaw > 0.0f) {
            float attScore = attention->GetScore(context, token);
            if (attScore < 0.0f) {
                attScore = 0.0f;
            }
            attNorm = static_cast<double>(attScore) /
                      static_cast<double>(maxAttRaw);
            if (attNorm < 0.0) {
                attNorm = 0.0;
            }
            if (attNorm > 1.0) {
                attNorm = 1.0;
            }
        }

        // Embedding similarity normalized to [0, 1].
        double embNorm = 0.0;
        if (wEmb > 0.0 && haveContextEmbedding && haveEmbRange &&
            (maxEmbRaw > minEmbRaw)) {

            const Embedding* tokenEmb = embedding->GetEmbeddingPtr(token);
            if (tokenEmb != NULL) {
                float dot = 0.0f;
                float normToken = 0.0f;
                int d;
                for (d = 0; d < EMBEDDING_WIDTH; ++d) {
                    float tv = tokenEmb->v[d];
                    dot       += contextEmbedding.v[d] * tv;
                    normToken += tv * tv;
                }

                if (normToken > 0.0f) {
                    float tokenNorm = std::sqrt(normToken);
                    if (tokenNorm > 0.0f) {
                        float sim = dot / tokenNorm;

                        double simD = static_cast<double>(sim);
                        embNorm = (simD - minEmbRaw) / (maxEmbRaw - minEmbRaw);
                        if (embNorm < 0.0) {
                            embNorm = 0.0;
                        }
                        if (embNorm > 1.0) {
                            embNorm = 1.0;
                        }
                    }
                }
            }
        }

        // Combine all three signals.
        double combined =
            wBase * baseNorm +
            wAtt  * attNorm  +
            wEmb  * embNorm;

        if (combined <= 0.0) {
            continue;
        }

        // Temperature shaping: combined in (0,1], raise to 1/T.
        double exponent = 1.0 / static_cast<double>(effectiveTemp);
        double weight   = std::pow(combined, exponent);

        if (weight <= 0.0) {
            continue;
        }

        tokens.push_back(token);
        weights.push_back(weight);
        totalWeight += weight;
    }

    // If numerical issues kill all weights, fall back to uniform.
    if (tokens.empty() || totalWeight <= 0.0) {
        tokens.clear();
        weights.clear();
        totalWeight = 0.0;

        for (it = baseScores.begin(); it != baseScores.end(); ++it) {
            tokens.push_back(it->first);
            weights.push_back(1.0);
            totalWeight += 1.0;
        }
    }
}





int SamplerSystem::SampleFromDistribution(
    const std::vector<int>& tokens,
    const std::vector<double>& weights,
    double totalWeight) const
{
    if (tokens.empty()) {
        return -1;
    }

    if (totalWeight <= 0.0) {
        int idx = std::rand() % static_cast<int>(tokens.size());
        return tokens[static_cast<std::size_t>(idx)];
    }

    double r = static_cast<double>(std::rand()) /
               static_cast<double>(RAND_MAX);
    double target = r * totalWeight;

    double accum = 0.0;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        accum += weights[i];
        if (accum >= target) {
            return tokens[i];
        }
    }

    return tokens.back();
}

// -----------------------------------------------------------------------------
// Main sampler entry point
// -----------------------------------------------------------------------------

int SamplerSystem::SampleNextToken(const std::vector<int>& context,
                                   const std::vector<std::vector<int>>& focus,
                                   const SamplerParameters& params) {
    // Basic sanity checks (unchanged)
    if (context.empty()) {
        return -2; // context empty
    }
    if (focus.empty()) {
        return -3; // focus empty
    }

    const int contextSize     = static_cast<int>(context.size());
    const int maxSentenceLen  = 32;
    const int sentenceStart   = GetSentenceStart(contextSize, maxSentenceLen);

    // PASS 1: best match length per span + global best
    std::vector<int> spanBestLen;
    int globalBestLen  = 0;
    int globalBestSpan = -1;

    ComputeSpanBestMatches(context,
                           sentenceStart,
                           maxSentenceLen,
                           focus,
                           spanBestLen,
                           globalBestLen,
                           globalBestSpan);

    // PASS 2: build score maps (locked vs all)
    std::unordered_map<int, double> lockedScores;
    std::unordered_map<int, double> allScores;

    BuildScoreMaps(context,
                   sentenceStart,
                   maxSentenceLen,
                   focus,
                   spanBestLen,
                   globalBestLen,
                   globalBestSpan,
                   lockedScores,
                   allScores);

    // Fallback if no matches at all
    FallbackToFrequencyScores(focus, allScores, globalBestLen);

    if (allScores.empty()) {
        // Nothing to choose from.
        return -1;
    }

    // Decide whether to use lockedScores or allScores and what temperature
    bool useLockedScores = false;
    float effectiveTemp  = params.temperatureHigh;

    ChooseScoreSource(globalBestLen,
                      lockedScores,
                      allScores,
                      params,
                      useLockedScores,
                      effectiveTemp);

    const std::unordered_map<int, double>& chosenScores =
        useLockedScores ? lockedScores : allScores;

    // Build token distribution with attention / (future) embedding
    std::vector<int>    tokens;
    std::vector<double> weights;
    double              totalWeight = 0.0;

    BuildTokenDistribution(context,
                           chosenScores,
                           params,
                           effectiveTemp,
                           tokens,
                           weights,
                           totalWeight);

    // Finally sample a token from the distribution
    return SampleFromDistribution(tokens, weights, totalWeight);
}


TokenDistribution SamplerSystem::SampleNextTokenDistribution(const std::vector<int>& context,
                                                             const std::vector<std::vector<int>>& focus,
                                                             const SamplerParameters& params, int topk) {
    TokenDistribution dist;
    // Basic sanity checks (unchanged)
    if (context.empty()) {
        return dist; // context empty
    }
    if (focus.empty()) {
        return dist; // focus empty
    }
    
    const int contextSize     = static_cast<int>(context.size());
    const int maxSentenceLen  = 32;
    const int sentenceStart   = GetSentenceStart(contextSize, maxSentenceLen);

    // PASS 1: best match length per span + global best
    std::vector<int> spanBestLen;
    int globalBestLen  = 0;
    int globalBestSpan = -1;

    ComputeSpanBestMatches(context,
                           sentenceStart,
                           maxSentenceLen,
                           focus,
                           spanBestLen,
                           globalBestLen,
                           globalBestSpan);

    // PASS 2: build score maps (locked vs all)
    std::unordered_map<int, double> lockedScores;
    std::unordered_map<int, double> allScores;

    BuildScoreMaps(context,
                   sentenceStart,
                   maxSentenceLen,
                   focus,
                   spanBestLen,
                   globalBestLen,
                   globalBestSpan,
                   lockedScores,
                   allScores);

    // Fallback if no matches at all
    FallbackToFrequencyScores(focus, allScores, globalBestLen);

    if (allScores.empty()) {
        // Nothing to choose from.
        return dist;
    }

    // Decide whether to use lockedScores or allScores and what temperature
    bool useLockedScores = false;
    float effectiveTemp  = params.temperatureHigh;

    ChooseScoreSource(globalBestLen,
                      lockedScores,
                      allScores,
                      params,
                      useLockedScores,
                      effectiveTemp);

    const std::unordered_map<int, double>& chosenScores =
        useLockedScores ? lockedScores : allScores;

    // Build token distribution with attention / (future) embedding
    std::vector<int>    tokens;
    std::vector<double> weights;
    double              totalWeight = 0.0;

    BuildTokenDistribution(context,
                           chosenScores,
                           params,
                           effectiveTemp,
                           tokens,
                           weights,
                           totalWeight);
    // ---- NEW: pick top-K highest probabilities ----
    if (tokens.empty()) {
        return dist;
    }

    // If BuildTokenDistribution did not set totalWeight, compute it here
    if (totalWeight <= 0.0) {
        for (std::size_t i = 0; i < weights.size(); ++i) {
            totalWeight += weights[i];
        }
    }

    if (totalWeight <= 0.0) {
        // Degenerate case, nothing with positive weight.
        return dist;
    }

    // Normalize to probabilities and pair up with tokens
    std::vector<std::pair<int, double> > tokenProbs;
    tokenProbs.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        double p = weights[i] / totalWeight;
        if (p > 0.0) {
            tokenProbs.push_back(std::make_pair(tokens[i], p));
        }
    }

    if (tokenProbs.empty()) {
        return dist;
    }

    // Sort by probability descending
    std::sort(tokenProbs.begin(),
              tokenProbs.end(),
              [](const std::pair<int, double>& a,
                 const std::pair<int, double>& b) {
                  return a.second > b.second;
              });

    // Take the top K
    std::size_t limit = tokenProbs.size();
    if (limit > topk) {
        limit = topk;
    }

    dist.tokens.reserve(limit);
    dist.weights.reserve(limit);
    for (std::size_t i = 0; i < limit; ++i) {
        dist.tokens.push_back(tokenProbs[i].first);
        dist.weights.push_back(tokenProbs[i].second); // already normalized probs
    }

    return dist;
}
