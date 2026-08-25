#include <stdio.h>
void printBinary(int value) {
    unsigned int bitValue = (unsigned int)value;
    for (int i = 31; i >= 0; i--) {
        printf("%d", (bitValue >> i) & 1u);
    }
}

int main(void) {
    int n;
    int sum = 0;
    int checksum;

    printf("Enter number of data blocks: ");
    scanf("%d", &n);

    if (n <= 0) {
        printf("Invalid number of data blocks.\n");
        return 1;
    }

    int data[n];

    printf("Enter %d data values:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &data[i]);
        sum += data[i];
    }

    checksum = ~sum;

    printf("\nSum = %d\n", sum);
    printf("Sum in binary (1s and 0s) = ");
    printBinary(sum);
    printf("\n");

    printf("Calculated Checksum = %d\n", checksum);
    printf("Checksum in binary (1s and 0s) = ");
    printBinary(checksum);
    printf("\n");

    printf("\nData sent to receiver: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", data[i]);
    }
    printf("%d\n", checksum);

    printf("\nBinary form of each data value:\n");
    for (int i = 0; i < n; i++) {
        printf("Data[%d] = %d -> ", i, data[i]);
        printBinary(data[i]);
        printf("\n");
    }

    return 0;
}