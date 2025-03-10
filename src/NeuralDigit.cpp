#include "data.h"
#include "nn.h"
#include <iostream>
#include <fstream>
using namespace std;

const int c_numInputNeurons = 785;
const int c_numHiddenNeurons = 30;
const int c_numOutputNeurons = 10;

const int c_trainingEpochs = 30;
const int c_miniBatchSize = 10;
const float c_learningRate = 3;

TrainingData g_trainingData;
TrainingData g_testData;

NeuralNetwork<c_numInputNeurons, c_numHiddenNeurons, c_numOutputNeurons> g_neuralNetwork;

float GetDataAccuracy(const TrainingData &data)
{
    int correctItems = 0;
    int c = data.NumImages();
    for (int i = 0; i < c; ++i)
    {
        uint8_t label;
        const float *pixels = data.GetImage(i, label);
        uint8_t detectedLabel = g_neuralNetwork.ForwardPass(pixels, label);
        if (detectedLabel == label)
            ++correctItems;
    }
    return float(correctItems) / float(data.NumImages());
}

int main()
{
    
    if (!g_trainingData.Load(true) || !g_testData.Load(false))
    {
        cout << "FAILED" << endl;
        return 1;
    }

    for (int epoch = 0; epoch < c_trainingEpochs; ++epoch)
    {
        float accuracyTraining = GetDataAccuracy(g_trainingData);
        float accuracyTest = GetDataAccuracy(g_testData);
        cout << "Training data accuracy: " << 100 * accuracyTraining << "%" << endl;
        cout << "Test data accuracy: " << 100 * accuracyTest << "%" << endl;
        cout << "Training the epoch " << epoch + 1 << " / " << c_trainingEpochs << "..." << endl;
        g_neuralNetwork.Train(g_trainingData, c_miniBatchSize, c_learningRate);
        cout << endl;
    }

    float accuracyTraining = GetDataAccuracy(g_trainingData);
    float accuracyTest = GetDataAccuracy(g_testData);
    cout << "\nFinal training data accuracy: " << 100 * accuracyTraining << "%" << endl;
    cout << "Final test data accuracy: " << 100 * accuracyTest << "%" << endl;

    ofstream json("Weights.txt");
    json << "{\n";
    json << "  \"InputNeurons\":" << c_numInputNeurons << ",\n";
    json << "  \"HiddenNeurons\":" << c_numHiddenNeurons << ",\n";
    json << "  \"OutputNeurons\":" << c_numOutputNeurons << ",\n";

    auto hiddenBiases = g_neuralNetwork.GetHiddenLayerBiases();
    json << "  \"HiddenBiases\" : [\n";
    for (int i = 0; i < hiddenBiases.size(); ++i)
    {
        json << "    " << hiddenBiases[i];
        if (i < hiddenBiases.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ],\n";

    auto hiddenWeights = g_neuralNetwork.GetHiddenLayerWeights();
    json << "  \"HiddenWeights\" : [\n";
    for (int i = 0; i < hiddenWeights.size(); ++i)
    {
        json << "    " << hiddenWeights[i];
        if (i < hiddenWeights.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ],\n";

    auto outputBiases = g_neuralNetwork.GetOutputLayerBiases();
    json << "  \"OutputBiases\" : [\n";
    for (int i = 0; i < outputBiases.size(); ++i)
    {
        json << "    " << outputBiases[i];
        if (i < outputBiases.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ],\n";

    auto outputWeights = g_neuralNetwork.GetOutputLayerWeights();
    json << "  \"OutputWeights\" : [\n";
    for (int i = 0; i < outputWeights.size(); ++i)
    {
        json << "    " << outputWeights[i];
        if (i < outputWeights.size() - 1)
            json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return 0;
}