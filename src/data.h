#include <cstdio>
#include <cstdint>
#include <vector>
using namespace std;

uint32_t EndianSwap(uint32_t a)
{
    return (a << 24) | ((a << 8) & 0x00ff0000) |
           ((a >> 8) & 0x0000ff00) | (a >> 24);
}

class TrainingData
{
private:
    uint8_t *dataLabel;
    uint8_t *dataImage;
    int imageCount;
    uint8_t *labels;
    uint8_t *pixels;
    vector<float> floatPixels;

public:
    TrainingData()
    {
        dataLabel = nullptr;
        dataImage = nullptr;
        imageCount = 0;
        labels = nullptr;
        pixels = nullptr;
    }

    bool Load(bool training)
    {
        imageCount = training ? 60000 : 10000;

        const char *labelsFileName = training ? "../data/train-labels.idx1-ubyte" : "../data/t10k-labels.idx1-ubyte";
        FILE *file = fopen(labelsFileName, "rb");
        if (!file)
        {
            printf("Could not open %s for reading.\n", labelsFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        dataLabel = new uint8_t[fileSize];
        fread(dataLabel, fileSize, 1, file);
        fclose(file);

        const char *imagesFileName = training ? "../data/train-images.idx3-ubyte" : "../data/t10k-images.idx3-ubyte";
        file = fopen(imagesFileName, "rb");
        if (!file)
        {
            printf("Could not open %s for reading.\n", imagesFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        dataImage = new uint8_t[fileSize];
        fread(dataImage, fileSize, 1, file);
        fclose(file);

        uint32_t *data = (uint32_t *)dataLabel;
        if (data[0] == 0x01080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
        }

        if (data[0] != 2049 || data[1] != imageCount)
        {
            printf("The label data contains unexpected header values.\n");
            return false;
        }
        labels = (uint8_t *)&(data[2]);

        data = (uint32_t *)dataImage;
        if (data[0] == 0x03080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
            data[2] = EndianSwap(data[2]);
            data[3] = EndianSwap(data[3]);
        }

        if (data[0] != 2051 || data[1] != imageCount || data[2] != 28 || data[3] != 28)
        {
            printf("The label data contains unexpected header values.\n");
            return false;
        }
        pixels = (uint8_t *)&(data[4]);

        floatPixels.resize(imageCount * 28 * 28);
        for (size_t i = 0; i < 28 * 28 * imageCount; ++i)
            floatPixels[i] = float(pixels[i]) / 255.0f;

        return true;
    }

    ~TrainingData()
    {
        delete[] dataLabel;
        delete[] dataImage;
    }

    size_t NumImages() const { return imageCount; }

    const float *GetImage(size_t index, uint8_t &label) const
    {
        label = labels[index];
        return &floatPixels[index * 28 * 28];
    }
};