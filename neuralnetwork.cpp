#include <iostream>
#include<cmath>
using namespace std;

int main() {

    int layerCount = 3;
    int* layerSizes = new int[layerCount - 1];
    layerSizes[0] = 14;
    layerSizes[1] = 16;
    layerSizes[2] = 2;

    


    return 0;
}

class ArrayNetwork {
    private:
        int layerCount;
        int* layerSizes;
        int*** network;
        int* inputActivations;
    public:
        ArrayNetwork(int& layerCount, int*& layerSizes);
        void computeInput();
        int getLayerCount();
        int getLayerSize(int layerIdx);
        double getActivation(int layerIdx, int neuronIdx);
        double getBias(int layerIdx, int neuronIdx);
        double getWeight(int layerIdx, int neuronIdx, int prevNeuronIdx);
        void setActivation(int layerIdx, int neuronIdx, double activation);
        void setBias(int layerIdx, int neuronIdx, double bias);
        void setWeight(int layerIdx, int neuronIdx, int prevNeuronIdx, double weight);
};

ArrayNetwork::ArrayNetwork(int& layerCount, int*& layerSizes): layerCount(layerCount), layerSizes(layerSizes) {
    
    // DATA!!!
    int*** network = new int**[layerCount - 1];

    // Initialize (what seems to be) layer 0
    int* inputActivations = new int[layerSizes[0]];

    for (int layer = 1; layer < layerCount; layer++) {
        network[layer] = new int*[layerSizes[layer]];
        for (int neuron = 0; neuron < layerSizes[layer]; neuron++) {
            
            // Create neuron
            network[layer][neuron] = new int[layerSizes[layer - 1] + 2];

            // Activation
            network[layer][neuron][0] = 0;

            // Bias
            network[layer][neuron][1] = 0;

            // Weights
            for (int weight = 2; weight < layerSizes[layerSizes[layer - 1] + 2]; weight++) {
                network[layer][neuron][2] = 0;
            }
        }
    }
}

void ArrayNetwork::computeInput() {
    int activation;
    for (int layerIdx = 1; layerIdx < layerCount; layerIdx++) {
        for (int neuronIdx = 0; neuronIdx < layerSizes[layerIdx]; neuronIdx++) {
            activation = getBias(layerIdx, neuronIdx);
            for (int prevNeuronIdx = getLayerSize(layerIdx - 1); prevNeuronIdx >= 0; prevNeuronIdx) {
                activation += getWeight(layerIdx, neuronIdx, prevNeuronIdx) * getActivation(layerIdx - 1, prevNeuronIdx);
            }
            activation = 1 / (1 + exp(activation));
            setActivation(layerIdx, neuronIdx, activation);
        }
    }
}

int ArrayNetwork::getLayerCount() {
    return layerCount;
}

int ArrayNetwork::getLayerSize(int layerIdx) {
    if (0 <= layerIdx && layerIdx < layerCount) {
        return layerSizes[layerIdx];
    } else {
        throw std::invalid_argument("layerIdx was out of bounds");
    }
}

double ArrayNetwork::getActivation(int layerIdx, int neuronIdx) {
    if (0 <= layerIdx && layerIdx < layerCount) {
        if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx]) {
            return layerIdx == 0
                ? inputActivations[neuronIdx]
                : network[layerIdx - 1][neuronIdx][0];
        } else {
            throw std::invalid_argument("neuronIdx was out of bounds");
        }
    } else {
        throw std::invalid_argument("layerIdx was out of bounds");
    }
}

double ArrayNetwork::getBias(int layerIdx, int neuronIdx) {
    if (0 < layerIdx && layerIdx < layerCount) {
        if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx]) {
            return network[layerIdx - 1][neuronIdx][1];
        } else {
            throw std::invalid_argument("neuronIdx was out of bounds");
        }
    } else {
        throw std::invalid_argument("layerIdx was out of bounds. The input layer doesn't have a bias.");
    }
}

double ArrayNetwork::getWeight(int layerIdx, int neuronIdx, int prevNeuronIdx) {
    if (0 < layerIdx && layerIdx < layerCount) {
        if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx]) {
            if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx - 1]) {
                return network[layerIdx - 1][neuronIdx][2 + prevNeuronIdx];
            } else {
                throw std::invalid_argument("prevNeuronIdx was out of bounds");
            }
        } else {
            throw std::invalid_argument("neuronIdx was out of bounds");
        }
    } else {
        throw std::invalid_argument("layerIdx was out of bounds. The input layer doesn't have weights.");
    }
}

void ArrayNetwork::setActivation(int layerIdx, int neuronIdx, double activation) {
    if (0 <= layerIdx && layerIdx < layerCount) {
        if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx]) {
            if (layerIdx == 0) {
                inputActivations[neuronIdx] = activation;
            } else {
                network[layerIdx - 1][neuronIdx][0] = activation;
            }
        } else {
            throw std::invalid_argument("neuronIdx was out of bounds");
        }
    } else {
        throw std::invalid_argument("layerIdx was out of bounds");
    }
}

void ArrayNetwork::setBias(int layerIdx, int neuronIdx, double bias) {
    if (0 < layerIdx && layerIdx < layerCount) {
        if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx]) {
            network[layerIdx - 1][neuronIdx][1] = bias;
        } else {
            throw std::invalid_argument("neuronIdx was out of bounds");
        }
    } else {
        throw std::invalid_argument("layerIdx was out of bounds. The input layer doesn't have a bias.");
    }
}

void ArrayNetwork::setWeight(int layerIdx, int neuronIdx, int prevNeuronIdx, double weight) {
    if (0 < layerIdx && layerIdx < layerCount) {
        if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx]) {
            if (0 <= neuronIdx && neuronIdx < layerSizes[layerIdx - 1]) {
                network[layerIdx - 1][neuronIdx][2 + prevNeuronIdx] = weight;
            } else {
                throw std::invalid_argument("prevNeuronIdx was out of bounds");
            }
        } else {
            throw std::invalid_argument("neuronIdx was out of bounds");
        }
    } else {
        throw std::invalid_argument("layerIdx was out of bounds. The input layer doesn't have weights.");
    }
}
