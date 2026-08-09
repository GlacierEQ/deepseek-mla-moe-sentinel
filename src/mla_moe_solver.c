/**
 * Local low-rank projection and top-K routing reference.
 *
 * This file exercises fixed-dimension matrix arithmetic and routing mechanics.
 * It does not establish fidelity to DeepSeek V3/R1, auxiliary-loss-free training,
 * production MLA/MoE behavior, or measured model performance.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define HIDDEN_DIM 512
#define LATENT_DIM 64
#define NUM_EXPERTS 8
#define TOP_K 2

typedef struct {
    float projection_kv[HIDDEN_DIM * LATENT_DIM];
    float projection_q[HIDDEN_DIM * LATENT_DIM];
    float expert_router[HIDDEN_DIM * NUM_EXPERTS];
} MLAMoEWeights;

typedef struct {
    int expert_indices[TOP_K];
    float expert_weights[TOP_K];
} MoERoutingDecision;

/** Project a fixed-size hidden vector into a local latent representation. */
void mla_compress_vector(
    const float* hidden,
    const float* projection_matrix,
    float* latent_out
) {
    for (int l = 0; l < LATENT_DIM; l++) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_DIM; h++) {
            sum += hidden[h] * projection_matrix[h * LATENT_DIM + l];
        }
        latent_out[l] = sum;
    }
}

/** Select local top-K experts from fixed router logits and normalize their weights. */
MoERoutingDecision moe_route(const float* hidden, const float* router_matrix) {
    float logits[NUM_EXPERTS];

    for (int e = 0; e < NUM_EXPERTS; e++) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_DIM; h++) {
            sum += hidden[h] * router_matrix[h * NUM_EXPERTS + e];
        }
        logits[e] = sum;
    }

    MoERoutingDecision decision;
    for (int k = 0; k < TOP_K; k++) {
        float max_val = -INFINITY;
        int max_idx = -1;
        for (int e = 0; e < NUM_EXPERTS; e++) {
            bool chosen = false;
            for (int prev = 0; prev < k; prev++) {
                if (decision.expert_indices[prev] == e) {
                    chosen = true;
                    break;
                }
            }
            if (!chosen && logits[e] > max_val) {
                max_val = logits[e];
                max_idx = e;
            }
        }
        decision.expert_indices[k] = max_idx;
        decision.expert_weights[k] = max_val;
    }

    float max_logit = decision.expert_weights[0];
    float sum_exp = 0.0f;
    for (int k = 0; k < TOP_K; k++) {
        decision.expert_weights[k] = expf(decision.expert_weights[k] - max_logit);
        sum_exp += decision.expert_weights[k];
    }
    for (int k = 0; k < TOP_K; k++) {
        decision.expert_weights[k] /= sum_exp;
    }

    return decision;
}

int main(void) {
    float hidden[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = 0.1f * (i % 10);
    }

    float projection[HIDDEN_DIM * LATENT_DIM];
    for (int i = 0; i < HIDDEN_DIM * LATENT_DIM; i++) {
        projection[i] = 0.01f;
    }

    float latent[LATENT_DIM];
    mla_compress_vector(hidden, projection, latent);

    printf("[MLA-reference] 512-dim to 64-dim latent sample[0]=%.4f\n", latent[0]);
    return 0;
}
