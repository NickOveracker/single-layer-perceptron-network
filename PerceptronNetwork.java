/**
 * Single-layer perceptron network with supervised learning.
 *
 * USAGE: java PerceptronNetwork <numInputs> <numOutputs> <trainingFile> <inputFile>
 *
 * Expected training input file format:
 *  -   First line: Integer denoting number of training vectors.
 *  -   1 training vector per line
 *  -   Inputs followed by expected outputs
 *  -   Vector elements are space-delimited
 *
 * Expected test input file format:
 *  -   First line: Integer denoting number of test vectors.
 *  -   1 test vector per line
 *  -   Inputs only (no expected outputs)
 *  -   Vector elements are space-delimited
 */
class PerceptronNetwork
{
    private static java.io.File trainingDataFile;
    private static java.io.File testDataFile;
    private static int numInputs;
    private static int numNeurons;
    private static int[] biases;
    private static double[][] weights;
    private static double[][] trainingData;
    private static double[][] testData;
    

    public static void main(String[] args)
    {
        // Parse the input, show proper usage and exit if incorrectly formatted.
        try
        {
            numInputs        = Integer.parseInt(args[0]);
            numNeurons       = Integer.parseInt(args[1]);
            trainingDataFile = new java.io.File(args[2]);
            testDataFile     = new java.io.File(args[3]);
        }
        catch(Exception e)
        {
            System.out.println("USAGE: java PerceptronNetwork <numInputs> <numOutputs> <trainingFile> <inputFile>");
            System.exit(1);
        }

        // Initialize synapses. All synapse weights default to zero.
        biases  = new int[numNeurons];
        weights = new double[numNeurons][numInputs];

        // Read in the input data to static double[][] inputData
        parseInputData();

        // Train the network
        boolean doneTraining = false;
        do
        {
            doneTraining = trainNetwork();
        } while(!doneTraining);

        // Print the input weight matrix
        System.out.println("Weights:");

        for(int i = 0; i < weights.length; i++)
        {
            for(int j = 0; j < weights[i].length; j++)
            {
                System.out.print(weights[i][j] + "\t");
            }
            System.out.println();
        }

        // Print the bias weight vector
        System.out.println("Biases:");

        for(int i = 0; i < biases.length; i++)
        {
            System.out.println(biases[i] + "\t");
        }

        // Feed the test input to the network, and print the outputs.
        for(int i = 0; i < testData.length; i++)
        {
            System.out.print("Input Vector " + (i + 1) + ": ");
            System.out.print(vectorToString(testData[i]));
            System.out.print("^T -- Output: ");
            System.out.print(vectorToString(classify(testData[i])));
            System.out.println("^T");
        }
    }

    /**
     * Return the calculated output of the neural network for a given input.
     *
     * @param inputVector The input vector.
     * @return An array of each neuron's output.
     */
    private static double[] classify(double[] inputVector)
    {
        double[] output = new double[numNeurons];
        double dotProduct;

        for(int neur = 0; neur < numNeurons; neur++)
        {
            // Perform a dot product of the current row of the weight matrix
            // with the input vector.
            dotProduct = 0;
            for(int p = 0; p < numInputs; p++)
            {
                dotProduct += weights[neur][p] * inputVector[p];
            }
            output[neur] = dotProduct + biases[neur] >= 0 ? 1 : 0;
        }

        return output;
    }

    /**
     * Double vector-to-string converter.
     *
     * @param vector An array of any doubles to be converted to a
     *               vector string.
     * @return A String beginning with '[ ', followed by each element of
     *         the input array, each followed by exactly one space, and
     *         terminated with ']'.
     */
    private static String vectorToString(double[] vector)
    {
        String ret = "[ ";

        for(int i = 0; i < vector.length; i++)
        {
            ret += vector[i] + " ";
        }

        return ret + "]";
    }

    /**
     * Use the perceptron training rule to train the network.
     *
     * @return True if any changes were made to the synapses during execution.
     **/
    private static boolean trainNetwork()
    {
        boolean noChanges = true;
        double dotProduct;
        int output;
        int error; // error = expected output - actual output

        // Go through every row of the weight matrix
        for(int line = 0; line < trainingData.length; line++)
        {
            for(int neur = 0; neur < numNeurons; neur++)
            {
                // Perform a dot product of the current row of the weight matrix
                // with the input vector.
                dotProduct = 0;
                for(int p = 0; p < numInputs; p++)
                {
                    dotProduct += weights[neur][p] * trainingData[line][p];
                }
                output = dotProduct + biases[neur] >= 0 ? 1 : 0;
                error = (int) trainingData[line][numInputs + neur] - output;

                // Adjust weights of synapses as needed.
                if(error != 0)
                {
                    noChanges = false;
                    biases[neur] += error;
                    for(int p = 0; p < numInputs; p++)
                    {
                        weights[neur][p] += error*trainingData[line][p];
                    }
                }
            }
        }

        return noChanges;
    }

    /**
     * Read the training data and the test data into the program.
     */
    private static void parseInputData()
    {
        char[] trainingDataArray  = new char[(int) (trainingDataFile.length())];
        char[] testDataArray      = new char[(int)     (testDataFile.length())];

        // Read in both files.
        try
        {
            (new java.io.FileReader(trainingDataFile)).read(trainingDataArray);
            (new java.io.FileReader(testDataFile    )).read(testDataArray);
        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(1);
        }

        // Parse the training data.
        String inputDataString = new String(trainingDataArray);
        java.util.Scanner scan = new java.util.Scanner(inputDataString);
        int inputDataLength    = scan.nextInt();
        trainingData           = new double[inputDataLength][numInputs + numNeurons];

        for(int i = 0; i < trainingData.length; i++)
        {
            for(int j = 0; j < trainingData[i].length; j++)
            {
                trainingData[i][j] = scan.nextDouble();
            }
        }

        // Parse the test data.
        inputDataString = new String(testDataArray);
        scan            = new java.util.Scanner(inputDataString);
        inputDataLength = scan.nextInt();
        testData        = new double[inputDataLength][numInputs];

        for(int i = 0; i < testData.length; i++)
        {
            for(int j = 0; j < testData[i].length; j++)
            {
                testData[i][j] = scan.nextDouble();
            }
        }

    }
}
