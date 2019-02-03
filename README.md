# single-layer-perceptron-network
Not intended for serious use. I am following Hagan et al.'s "Neural Network Design" book, but I don't have Matlab. So, I am writing my own code in Java as a substitute for their Matlab libraries for some of the exercises.

Single-layer perceptron network with supervised learning.

USAGE: java PerceptronNetwork <numInputs> <numOutputs> <trainingFile> <inputFile>

Expected training input file format:
-   First line: Integer denoting number of training vectors.
-   1 training vector per line
-   Inputs followed by expected outputs
-   Vector elements are space-delimited

Expected test input file format:
-   First line: Integer denoting number of test vectors.
-   1 test vector per line
-   Inputs only (no expected outputs)
-   Vector elements are space-delimited
