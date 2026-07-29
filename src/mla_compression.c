#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double sequence_length;
    double head_dim;
    double num_heads;
} DeepSeekMLASpec;

double compute_kv_cache_compression(DeepSeekMLASpec spec) {
    double standard_kv_size = 2 * spec.sequence_length * spec.num_heads * spec.head_dim;
    double mla_compressed_size = spec.sequence_length * (spec.head_dim / 4.0); // 4x MLA compression
    return (1.0 - (mla_compressed_size / standard_kv_size)) * 100.0;
}

int main() {
    DeepSeekMLASpec spec = {1000000.0, 128.0, 128.0};
    double compression = compute_kv_cache_compression(spec);
    printf("DeepSeek MLA KV-Cache Compression Ratio: %.2f%%\n", compression);
    return 0;
}
