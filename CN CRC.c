#include <stdio.h>
#include <string.h>

void bitStuffing(char data[], char stuffedData[]) {
    int i = 0, j = 0;
    int count = 0;

    while (data[i] != '\0') {
        if (data[i] == '1') {
            count++;
        } else {
            count = 0;
        }

        stuffedData[j++] = data[i];

        if (count == 5) {
            stuffedData[j++] = '0';
            count = 0;
        }
        i++;
    }
    stuffedData[j] = '\0';
}

void bitDestuffing(char stuffedData[], char destuffedData[]) {
    int i = 0, j = 0;
    int count = 0;

    while (stuffedData[i] != '\0') {
        if (stuffedData[i] == '1') {
            count++;
        } else {
            count = 0;
        }

        destuffedData[j++] = stuffedData[i];

        if (count == 5) {
            i++;
            count = 0;
        }
        i++;
    }
    destuffedData[j] = '\0';
}

int main() {
    char data[100], stuffedData[200], destuffedData[100];

    printf("Enter the binary data stream (e.g., 011010010): \n");
    scanf("%s", data);

    bitStuffing(data, stuffedData);
    printf("\n--- After Bit Stuffing ---\n");
    printf("Stuffed Data: %s\n", stuffedData);

    bitDestuffing(stuffedData, destuffedData);
    printf("\n--- After Bit Destuffing ---\n");
    printf("Destuffed Data: %s\n", destuffedData);

    return 0;
}