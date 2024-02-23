#include <stdio.h>
#include <stdlib.h>


typedef struct {
    float mean_x;
    float mean_y;
    float mean_z;
    // Add more features as needed
} AccFeatures;

void separate_axes(int16_t** x_data, int16_t** y_data, int16_t** z_data, size_t* total_length) {
    // Example data array
    int data[] = {-10,77,304,3,54,373,6,46,488,77,-114,511,-113,181,511,-9,-18,209,-68,-81,242,41,-88,66,-13,
    -55,135,-20,-19,169,-46,27,212,-64,51,275,-52,112,356,-51,208,449,-66,439,479,-131,498,284,-148,511,93,
    -117,445,222,415,-512,-288,160,505,194,-125,389,384,36,201,231,-27,78,245,0,93,237,-6,79,245,2,83,238,-27,
    75,240,-40,88,236,-3,96,235,-28,116,218,-37,148,209,-35,148,221,-15,159,288,-19,132,337,22,72,462,-37,-85,
    511,-72,167,511,-127,-2,59,101,-282,272,52,-88,141,97,-22,196,75,28,275,68,77,342,57,132,406,37,240,500,-60,
    495,466,-130,511,221,-159,511,123,-227,148,511,145,511,-196,101,-148,163,7,60,232,-30,80,246,-21,69,254,-12,
    62,241,-21,54,253,-16,50,246,-19,48,245,-14,57,244,-12,57,254};
    size_t data_length = sizeof(data) / sizeof(data[0]);

    // Calculate the number of samples (assuming data_length is divisible by 3)
    *total_length = data_length / 3;

    // Allocate memory for x, y, z arrays
    *x_data = (int16_t*)malloc(*total_length * sizeof(int16_t));
    *y_data = (int16_t*)malloc(*total_length * sizeof(int16_t));
    *z_data = (int16_t*)malloc(*total_length * sizeof(int16_t));

    // Check for successful allocations
    if (!*x_data || !*y_data || !*z_data) {
        // Handle allocation failure (e.g., by freeing already allocated memory and exiting)
        // For simplicity, this example will just print an error message and exit
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Separate the data into x, y, z components
    for (size_t i = 0, j = 0; i < data_length; i += 3, j++) {
        (*x_data)[j] = data[i];
        (*y_data)[j] = data[i + 1];
        (*z_data)[j] = data[i + 2];
    }
}

float calculate_mean(int16_t* data, size_t length) {
    float sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += data[i];
    }
    return sum / length;
}


AccFeatures calculate_features(int16_t* x_data, int16_t* y_data, int16_t* z_data, size_t length) {
    AccFeatures features;
    features.mean_x = calculate_mean(x_data, length);
    features.mean_y = calculate_mean(y_data, length);
    features.mean_z = calculate_mean(z_data, length);
    // Calculate and assign more features as needed
    return features;
}

int main() {
    int16_t *x_data, *y_data, *z_data;
    size_t total_length = 60;

    separate_axes(&x_data, &y_data, &z_data, &total_length);

    // Use x_data, y_data, and z_data...
    // For example, print the first few elements
    for (size_t i = 0; i < 5 && i < total_length; ++i) {
        printf("X: %d, Y: %d, Z: %d\n", x_data[i], y_data[i], z_data[i]);
    }



    AccFeatures features = calculate_features(x_data, y_data, z_data, total_length);

    // Example usage
    printf("Mean Acc X: %f\n", features.mean_x);
    printf("Mean Acc Y: %f\n", features.mean_y);
    printf("Mean Acc Z: %f\n", features.mean_z);








    // Remember to free the allocated memory when no longer needed
    free(x_data);
    free(y_data);
    free(z_data);

    return 0;
}
