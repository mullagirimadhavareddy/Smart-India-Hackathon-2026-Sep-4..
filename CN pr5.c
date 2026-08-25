#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int hammingDistanceStr(const char *a, const char *b) {
    int i, diff = 0;
    for (i = 0; a[i] && b[i]; ++i) {
        if (a[i] != b[i]) ++diff;
    }
    return diff;
}

int main(void) {
    int n, len;
    printf("Enter number of codewords and bit-length (e.g. '4 3'): ");
    if (scanf("%d %d", &n, &len) != 2 || n < 2 || len < 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }

    char **codes = malloc(n * sizeof(char*));
    if (!codes) return 2;

    for (int i = 0; i < n; ++i) {
        codes[i] = malloc(len + 2);
        if (!codes[i]) return 3;
        if (scanf("%s", codes[i]) != 1) {
            fprintf(stderr, "Failed to read codeword %d\n", i);
            return 4;
        }
        if ((int)strlen(codes[i]) != len) {
            fprintf(stderr, "Codeword '%s' has incorrect length (expected %d).\n", codes[i], len);
            return 5;
        }
    }

    int dmin = len + 1;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int d = hammingDistanceStr(codes[i], codes[j]);
            printf("d(%s, %s) = %d\n", codes[i], codes[j], d);
            if (d < dmin) dmin = d;
        }
    }

    if (dmin == len + 1) dmin = 0; /* only possible if n<2 but guarded earlier */
    printf("\nThe d_min in this set is %d.\n", dmin);

    for (int i = 0; i < n; ++i) free(codes[i]);
    free(codes);
    return 0;
}