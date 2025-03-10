#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
using namespace std;

template <int input_neurons, int hidden_neurons, int output_neurons>
class NeuralNetwork
{
private:
    array<float, input_neurons * hidden_neurons> hiddenLayerWeights;
    array<float, hidden_neurons * output_neurons> outputLayerWeights;

    array<float, hidden_neurons> hiddenLayerBiases;
    array<float, output_neurons> outputLayerBiases;

    array<float, hidden_neurons> hiddenLayerOutputs;
    array<float, output_neurons> outputLayerOutputs;

    array<float, hidden_neurons> hiddenBiasesDelta;
    array<float, output_neurons> outputBiasesDelta;

    array<float, input_neurons * hidden_neurons> hiddenWeightsDelta;
    array<float, hidden_neurons * output_neurons> outputWeightsDelta;

    array<float, hidden_neurons> batchHiddenBiasesDelta;
    array<float, output_neurons> batchOutputBiasesDelta;

    array<float, input_neurons * hidden_neurons> batchHiddenWeightsDelta;
    array<float, hidden_neurons * output_neurons> batchOutputWeightsDelta;

    vector<int> trainingOrder;

public:
    NeuralNetwork()
    {
        random_device rd;
        mt19937 e2(rd());
        normal_distribution<float> dist(0, 1);

        for (float &f : hiddenLayerBiases)
            f = dist(e2);

        for (float &f : outputLayerBiases)
            f = dist(e2);

        for (float &f : hiddenLayerWeights)
            f = dist(e2);

        for (float &f : outputLayerWeights)
            f = dist(e2);
    }

    void Train(const TrainingData &trainingData, int batchSize, float learningRate)
    {
        if (trainingOrder.size() != trainingData.NumImages())
        {
            trainingOrder.resize(trainingData.NumImages());
            int index = 0;
            for (int &v : trainingOrder)
            {
                v = index;
                ++index;
            }
        }
        static random_device rd;
        static mt19937 e2(rd());
        shuffle(trainingOrder.begin(), trainingOrder.end(), e2);

        int trainingIndex = 0;
        while (trainingIndex < trainingData.NumImages())
        {
            fill(batchHiddenBiasesDelta.begin(), batchHiddenBiasesDelta.end(), 0.0f);
            fill(batchOutputBiasesDelta.begin(), batchOutputBiasesDelta.end(), 0.0f);
            fill(batchHiddenWeightsDelta.begin(), batchHiddenWeightsDelta.end(), 0.0f);
            fill(batchOutputWeightsDelta.begin(), batchOutputWeightsDelta.end(), 0.0f);

            int miniBatchIndex = 0;
            while (miniBatchIndex < batchSize && trainingIndex < trainingData.NumImages())
            {
                uint8_t imageLabel = 0;
                const float *pixels = trainingData.GetImage(trainingOrder[trainingIndex], imageLabel);

                uint8_t labelDetected = ForwardPass(pixels, imageLabel);

                BackwardPass(pixels, imageLabel);

                for (int i = 0; i < hiddenBiasesDelta.size(); ++i)
                    batchHiddenBiasesDelta[i] += hiddenBiasesDelta[i];
                for (int i = 0; i < outputBiasesDelta.size(); ++i)
                    batchOutputBiasesDelta[i] += outputBiasesDelta[i];
                for (int i = 0; i < hiddenWeightsDelta.size(); ++i)
                    batchHiddenWeightsDelta[i] += hiddenWeightsDelta[i];
                for (int i = 0; i < outputWeightsDelta.size(); ++i)
                    batchOutputWeightsDelta[i] += outputWeightsDelta[i];

                ++trainingIndex;
                ++miniBatchIndex;
            }

            float miniBatchLearningRate = learningRate / float(miniBatchIndex);

            for (int i = 0; i < hiddenLayerBiases.size(); ++i)
                hiddenLayerBiases[i] -= batchHiddenBiasesDelta[i] * miniBatchLearningRate;
            for (int i = 0; i < outputLayerBiases.size(); ++i)
                outputLayerBiases[i] -= batchOutputBiasesDelta[i] * miniBatchLearningRate;
            for (int i = 0; i < hiddenLayerWeights.size(); ++i)
                hiddenLayerWeights[i] -= batchHiddenWeightsDelta[i] * miniBatchLearningRate;
            for (int i = 0; i < outputLayerWeights.size(); ++i)
                outputLayerWeights[i] -= batchOutputWeightsDelta[i] * miniBatchLearningRate;
        }
    }

    uint8_t ForwardPass(const float *pixels, uint8_t correctLabel)
    {
        for (int neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex)
        {
            float Z = hiddenLayerBiases[neuronIndex];

            for (int inputIndex = 0; inputIndex < input_neurons; ++inputIndex)
                Z += pixels[inputIndex] * hiddenLayerWeights[HiddenLayerWeightIndex(inputIndex, neuronIndex)];

            hiddenLayerOutputs[neuronIndex] = 1.0f / (1.0f + exp(-Z));
        }

        for (int neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex)
        {
            float Z = outputLayerBiases[neuronIndex];

            for (int inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
                Z += hiddenLayerOutputs[inputIndex] * outputLayerWeights[OutputLayerWeightIndex(inputIndex, neuronIndex)];

            outputLayerOutputs[neuronIndex] = 1.0f / (1.0f + exp(-Z));
        }

        float maxOutput = outputLayerOutputs[0];
        uint8_t maxLabel = 0;
        for (uint8_t neuronIndex = 1; neuronIndex < output_neurons; ++neuronIndex)
        {
            if (outputLayerOutputs[neuronIndex] > maxOutput)
            {
                maxOutput = outputLayerOutputs[neuronIndex];
                maxLabel = neuronIndex;
            }
        }
        return maxLabel;
    }

    void BackwardPass(const float *pixels, uint8_t correctLabel)
    {
        for (int neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex)
        {
            float desiredOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;

            float deltaCost_deltaO = outputLayerOutputs[neuronIndex] - desiredOutput;
            float deltaO_deltaZ = outputLayerOutputs[neuronIndex] * (1.0f - outputLayerOutputs[neuronIndex]);

            outputBiasesDelta[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

            for (int inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
                outputWeightsDelta[OutputLayerWeightIndex(inputIndex, neuronIndex)] = outputBiasesDelta[neuronIndex] * hiddenLayerOutputs[inputIndex];
        }

        for (int neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex)
        {
            float deltaCost_deltaO = 0.0f;
            for (int destinationNeuronIndex = 0; destinationNeuronIndex < output_neurons; ++destinationNeuronIndex)
                deltaCost_deltaO += outputBiasesDelta[destinationNeuronIndex] * outputLayerWeights[OutputLayerWeightIndex(neuronIndex, destinationNeuronIndex)];

            float deltaO_deltaZ = hiddenLayerOutputs[neuronIndex] * (1.0f - hiddenLayerOutputs[neuronIndex]);
            hiddenBiasesDelta[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

            for (int inputIndex = 0; inputIndex < input_neurons; ++inputIndex)
                hiddenWeightsDelta[HiddenLayerWeightIndex(inputIndex, neuronIndex)] = hiddenBiasesDelta[neuronIndex] * pixels[inputIndex];
        }
    }

    const array<float, hidden_neurons> &GetHiddenLayerBiases() const
    {
        return hiddenLayerBiases;
    }
    const array<float, output_neurons> &GetOutputLayerBiases() const
    {
        return outputLayerBiases;
    }
    const array<float, input_neurons * hidden_neurons> &GetHiddenLayerWeights() const
    {
        return hiddenLayerWeights;
    }
    const array<float, hidden_neurons * output_neurons> &GetOutputLayerWeights() const
    {
        return outputLayerWeights;
    }
    static int HiddenLayerWeightIndex(int inputIndex, int hiddenLayerNeuronIndex)
    {
        return hiddenLayerNeuronIndex * input_neurons + inputIndex;
    }
    static int OutputLayerWeightIndex(int hiddenLayerNeuronIndex, int outputLayerNeuronIndex)
    {
        return outputLayerNeuronIndex * hidden_neurons + hiddenLayerNeuronIndex;
    }
};