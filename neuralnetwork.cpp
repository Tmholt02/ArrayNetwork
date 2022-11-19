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
        const int layerCount;
        const int* layerSizes;
        int*** network;
        int* inputActivations;
    public:
        ArrayNetwork(const int& layerCount, const int* layerSizes);
        ~ArrayNetwork();
        void computeInput();
        int getLayerCount() const;
        int getLayerSize(const int& layerIdx) const;
        double getActivation(const int& layerIdx, const int& neuronIdx) const;
        double getBias(const int& layerIdx, const int& neuronIdx) const;
        double getWeight(const int& layerIdx, const int& neuronIdx, const int& prevNeuronIdx) const;
        void setActivation(const int& layerIdx, const int& neuronIdx, const double& activation);
        void setBias(const int& layerIdx, const int& neuronIdx, const double& bias);
        void setWeight(const int& layerIdx, const int& neuronIdx, const int& prevNeuronIdx, const double& weight);
};

ArrayNetwork::ArrayNetwork(const int& layerCount, const int* layerSizes): layerCount(layerCount), layerSizes(layerSizes) {
    
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
            for (int weight = 2; weight < layerSizes[layer - 1] + 2; weight++) {
                network[layer][neuron][weight] = 0;
            }
        }
    }
}

ArrayNetwork::~ArrayNetwork() {
    for (int layerIdx = 1; layerIdx < layerCount; layerIdx++) {
        for (int neuronIdx = 0; neuronIdx < layerSizes[layerIdx]; neuronIdx++) {
            delete[] network[layerIdx][neuronIdx];
        }
        delete[] network[layerIdx];
    }
    delete[] network;
    delete[] inputActivations;
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

int ArrayNetwork::getLayerCount() const {
    return layerCount;
}

int ArrayNetwork::getLayerSize(const int& layerIdx) const {
    if (0 <= layerIdx && layerIdx < layerCount) {
        return layerSizes[layerIdx];
    } else {
        throw std::invalid_argument("layerIdx was out of bounds");
    }
}

double ArrayNetwork::getActivation(const int& layerIdx, const int& neuronIdx) const {
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

double ArrayNetwork::getBias(const int& layerIdx, const int& neuronIdx) const {
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

double ArrayNetwork::getWeight(const int& layerIdx, const int& neuronIdx, const int& prevNeuronIdx) const {
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

void ArrayNetwork::setActivation(const int& layerIdx, const int& neuronIdx, const double& activation) {
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

void ArrayNetwork::setBias(const int& layerIdx, const int& neuronIdx, const double& bias) {
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

void ArrayNetwork::setWeight(const int& layerIdx, const int& neuronIdx, const int& prevNeuronIdx, const double& weight) {
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
