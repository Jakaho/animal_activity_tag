#include <stdio.h>
#include <stdlib.h>


typedef struct {
    float mean_x;
    float mean_y;
    float mean_z;

    int16_t max_x;
    int16_t max_y;
    int16_t max_z;

    int16_t min_x;
    int16_t min_y;
    int16_t min_z;

    int16_t Q5_x;
    int16_t Q5_y;
    int16_t Q5_z;

    int16_t Q95_x;
    int16_t Q95_y;
    int16_t Q95_z;

    // Add more features as needed
} AccFeatures;

void separate_axes(int16_t** x_data, int16_t** y_data, int16_t** z_data, size_t* total_length) {
    // Example data array
    int16_t data[] = {-10,77,304,3,54,373,6,46,488,77,-114,511,-113,181,511,-9,-18,209,-68,-81,242,41,-88,66,-13,
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

int calculate_max(int16_t* data, size_t length){
    int16_t current_max = 0;
    for (size_t i = 0; i < length; ++i){
        if(data[i] > current_max){
        current_max = data[i];
        }
    }
    return current_max;
}

int calculate_min(int16_t* data, size_t length){
    int16_t current_min = 1000;
    for (size_t i = 0; i < length; ++i){
        if(data[i] < current_min){
        current_min = data[i];
        }
    }
    return current_min;
}
void swap(int16_t* a, int16_t* b) {
    int16_t temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int16_t arr[], int low, int high) {
    int16_t pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

int16_t quickSelect(int16_t arr[], int low, int high, int k) {
    if (k > 0 && k <= high - low + 1) {
        int index = partition(arr, low, high);

        if (index - low == k - 1)
            return arr[index];
        if (index - low > k - 1)
            return quickSelect(arr, low, index - 1, k);

        return quickSelect(arr, index + 1, high, k - index + low - 1);
    }

    return INT16_MAX;
}

float* calculate_dynamic_components(int16_t* data, size_t length) {
    if(length < 2) return NULL; // Not enough data to compute differences

    float* dyn_components = (float*)malloc((length - 1) * sizeof(float));
    if (!dyn_components) {
        printf("Memory allocation failed for dynamic components\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < length - 1; i++) {
        dyn_components[i] = (float)(data[i + 1] - data[i]);
    }

    return dyn_components;
}



AccFeatures calculate_features(int16_t* x_data, int16_t* y_data, int16_t* z_data, size_t length) {
    AccFeatures features;
    int Q5 = 3;
    int Q95 = 58;
    features.mean_x = calculate_mean(x_data, length);
    features.mean_y = calculate_mean(y_data, length);
    features.mean_z = calculate_mean(z_data, length);

    features.max_x = calculate_max(x_data, length);
    features.max_y = calculate_max(y_data, length);
    features.max_z = calculate_max(z_data, length);

    features.min_x = calculate_min(x_data, length);
    features.min_y = calculate_min(y_data, length);
    features.min_z = calculate_min(z_data, length);

    features.Q5_x = quickSelect(x_data, 0, length, Q5);
    features.Q5_y = quickSelect(y_data, 0, length, Q5);
    features.Q5_z = quickSelect(z_data, 0, length, Q5);

    features.Q95_x = quickSelect(x_data, 0, length, Q95);
    features.Q95_y = quickSelect(y_data, 0, length, Q95);
    features.Q95_z = quickSelect(z_data, 0, length, Q95);

    // Calculate and assign more features as needed
    return features;
}

int main() {
    int16_t *x_data, *y_data, *z_data;
    size_t total_length = 60;
    separate_axes(&x_data, &y_data, &z_data, &total_length);

    AccFeatures features = calculate_features(x_data, y_data, z_data, total_length);


    // Calculate dynamic components for each axis, filters out steady/background acceleration like gravity. 
    float *dyn_x = calculate_dynamic_components(x_data, total_length);
    float *dyn_y = calculate_dynamic_components(y_data, total_length);
    float *dyn_z = calculate_dynamic_components(z_data, total_length);

    // Example usage of AC components (here we just print the first few for demonstration)
    printf("First few dynamic components for X-axis:\n");
    for (size_t i = 0; i < 5 && i < total_length - 1; i++) {
        printf("%.2f ", dyn_x[i]);
    }
    printf("\n");
   
    // Remember to free the allocated memory when no longer needed
    free(x_data);
    free(y_data);
    free(z_data);

    return 0;
}
