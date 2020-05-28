import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Scanner;

/**
 * @author Arjun Akkiraju
 * @version 1/17/2020
 * This class contains the methods and instance variables necessary to create a neural network
 * for a given number of input nodes, hidden layers (and the number of nodes in each
 * hidden layer), and output nodes, as well as a set of training parameters and test cases. These
 * values are inputted by the user via a configuration file, which is read and parsed using the
 * readConfigFile() method. For backpropagation, arrays of omegas, psis, and thetas are also
 * initialized, with each of these arrays having the same dimensions as the activations array.
 * Please note that there are two different modes for the neural network: training and running.
 * For the "train" mode, the user has the option to choose whether or not to randomize the
 * weights; if they choose not to, the weights would be manually input by the user via a given
 * weights file. The goal of training is to minimize the error for each test case. There are
 * three end conditions for the training:
 * End case 1: Maximum number of iterations are met
 * End case 2: Learning factor is less than learning factor threshold
 * End case 3: Error for each test case is less than the error threshold
 * The trainNetwork() has also been updated with the backpropagation algorithm, which has
 * decreased runtime significantly.
 * For more information, please view the description of the "trainNetwork()" method.
 * For the "run" mode, the network is run for each test case using weights that are specified by
 * the given weights file. Output values and errors are printed to the console for each test case.
 * The following are additional specifications/limitations of the network:
 * 1) The code currently only supports a three layer network. Although the code can
 * handle the front end of a network with more layers, the network is limited to only three layers
 * in terms of training and running.
 * 2) The activation function used for training and running purposes of the network is the
 * sigmoid function.
 *
 * Table of Contents:
 * public double[] makeActivationArray()
 * public void readConfigFile(String filepath)
 * public double[][][] makeTestSetsArray(String inputfilepath, String targetoutputsfilepath, int
 * numtestcases)
 * public double[][][] makeWeightsArray(String weightsfilepath)
 * public void loadInputs(int testcase)
 * public double generateRandomWeight(double lowerlim, double upperlim)
 * public double[][][] makeRandomWeightsArray()
 * public int findMaxNodesInLayer()
 * public double[] getOutputVals()
 * public double calculateError(int testcase)
 * public void runNetwork()
 * public void runMode(String weightsfilepath)
 * public void trainNetwork(boolean randomizeWeights, String weightsfilepath)
 * public void backProp(double learningfactor, int testcase)
 * public void writeWeightsToFile()
 * public double activationFunction(double dotproductsum)
 * public double derivativeActivationFunction(double dotproductsum)
 *
 */
public class NeuralNetwork
{
   private int inputnodes;
   private int[] hiddenLayerNodes;
   private int outputnodes;
   private double[][][] weights;
   private double[][] activations;
   private double[][][] testsets;
   private double[] trainingparams;

   private double[][] omegas;
   private double[][] psis;
   private double[][] thetas;

   private static final int NUMLINESCONFIG = 11; //the number of lines in the config file
   private static final int NUMTRAININGPARAMS = 6; // the number of parameters for training


   /**
    * Constructor for the NeuralNetwork class that initializes the trainingparams instance
    * variable.
    */
   public NeuralNetwork()
   {
      trainingparams = new double[NUMTRAININGPARAMS];
   }
   
   /**
    * Reads the config file line by line to determine all parameters necessary to train or run
    * the network. First, the program starts by parsing values necessary for both training and
    * running, specifically the number of input nodes, the number of output nodes, the number of
    * hidden layers, the number of nodes in each hidden layer, and the number of test cases.
    * Then, the six parameters required for training are parsed and stored in the trainingparams
    * array to be used when the network trains. Arrays needed for back propagation, namely
    * omegas, psis, and thetas, are initialized according to the dimensions of the
    * activations array. The user is prompted for files containing the inputs and target
    * outputs. Finally, the test cases are parsed one by one and stored in the "testsets" array
    * from the provided files, which contain the inputs and target outputs.
    * @param filepath the file path of the config file
    */
   public void readConfigFile(String filepath)
   {
      String configfilepath = filepath;

      try
      {
         String[] configvalues = new String[NUMLINESCONFIG];
         BufferedReader reader = new BufferedReader(new FileReader(configfilepath));

         String line = reader.readLine();
         int i = 0;

         while (i<configvalues.length&&line != null)
         {
            configvalues[i] = (line.split(":")[1]).substring(1);
            line = reader.readLine();
            i += 1;
         }

         inputnodes = Integer.parseInt(configvalues[0]);
         int numHiddenLayers = Integer.parseInt(configvalues[1]);
         hiddenLayerNodes = new int[numHiddenLayers];

         String[] numHiddenLayerNodes = configvalues[2].split(",");

         for (int j = 0; j < hiddenLayerNodes.length; j++)
         {
            hiddenLayerNodes[j] = Integer.parseInt(numHiddenLayerNodes[j]);
         }

         outputnodes = Integer.parseInt(configvalues[3]);
         activations = makeActivationArray();

         int activationlayers = activations.length;
         int maxnumnodes = activations[0].length;

         omegas = new double[activationlayers][maxnumnodes];
         psis = new double[activationlayers][maxnumnodes];
         thetas = new double[activationlayers][maxnumnodes];

         int numtestcases = Integer.parseInt(configvalues[4]);

         for (int j = 0; j < trainingparams.length; j++)
         {

            trainingparams[j] = Double.parseDouble(configvalues[j + 5]);

         }

         Scanner scanner = new Scanner(System.in);

         System.out.println("Please provide the file path of the file containing the inputs");
         String inputfilepath = scanner.next();
         if (inputfilepath.equals("."))
         {
            inputfilepath = "inputs.txt";
         }

         System.out.println("Please provide the file path of the file containing the target " +
                 "outputs");
         String targetoutputfilepath = scanner.next();

         if (targetoutputfilepath.equals("."))
         {
            targetoutputfilepath = "targetoutputs.txt";
         }

         double[][][] testcasevals = makeTestSetsArray(inputfilepath, targetoutputfilepath,
                 numtestcases);

         testsets = testcasevals;
      }

      catch (IOException e)
      {
         e.printStackTrace();
      }
   }
   /**
    * Initializes the activation array. The length of the activation array is set to be equal to
    * the number of hidden layers plus two to signify the inclusion of the input layer, hidden
    * layers, and output layers in the activation array. The width (length of the second
    * dimension) of the array is determined by using the helper function findMaxNumNodesInLayer,
    * which finds the maximum number of nodes in a layer of the network.
    * @return an array representing all of the activation nodes in the network
    */
   public double[][] makeActivationArray()
   {
      int activationlayers = hiddenLayerNodes.length + 2;
      int maxnumnodes = findMaxNodesInLayer();
      double[][] activationarray = new double[activationlayers][maxnumnodes];
      return activationarray;
   }

   /**
    * Creates the testsets array by reading in the inputs and target outputs for each test case
    * from the input file and target outputs file, respectively. Input and target output values
    * are separated by a single space in the files, and are read as such.
    * @param inputfilepath The path of the file containing the input values for each test case
    * @param targetoutputsfilepath The path of the file containing the target output values for
    *                              each test case
    * @param numtestcases The number of test cases
    * @return an array containing the inputs and target outputs for each test case
    */
   public double[][][] makeTestSetsArray(String inputfilepath, String targetoutputsfilepath,
                                         int numtestcases)
   {
      double[][][] testcasevals = new double[numtestcases][2][Math.max(inputnodes, outputnodes)];

      try
      {

         BufferedReader inputreader = new BufferedReader(new FileReader(inputfilepath));
         BufferedReader outputreader = new BufferedReader(new FileReader(targetoutputsfilepath));

         int i = 0;
         String[] inputvalues = new String[numtestcases];
         String[] outputvalues = new String[numtestcases];

         String inputline = " ";
         String outputline = " ";

         while (inputline!=null&&outputline!=null&&i<numtestcases)
         {
            inputline = inputreader.readLine();
            inputvalues[i] = inputline;

            outputline = outputreader.readLine();
            outputvalues[i] = outputline;

            i +=1;
         }

         for (int testcase = 0; testcase < numtestcases; testcase++)
         {
            String[] inputs = inputvalues[testcase].split(" ");

            for (int j = 0; j < inputnodes; j++)
            {
               testcasevals[testcase][0][j] = Double.parseDouble(inputs[j]);
            }

            String[] outputs = outputvalues[testcase].split(" ");

            for (int z = 0; z < outputnodes; z++)
            {

               testcasevals[testcase][1][z] = Double.parseDouble(outputs[z]);

            }
         } //for (int testcase = 0; testcase < numtestcases; testcase++)
         
         inputreader.close();
         outputreader.close();

      }
      
      catch (IOException e)
      {
         e.printStackTrace();
      }

      return testcasevals;
   }

   /**
    * Initializes the weights array and sets values for the weights according to user input from
    * the given weights file. Starts by initializing the weight array according to the maximum
    * number of nodes in a layer and the number of connectivity layers. Then, the file is read
    * line by line to get the indices of each weight along with its corresponding weight value
    * @param weightsfilepath the file path to the file containing the weights
    * @return an initialized weights array with values filled in according to the "weights.txt"
    * file
    */
   public double[][][] makeWeightsArray(String weightsfilepath)
   {
      int connectivitylayers = hiddenLayerNodes.length + 1;
      int weightarraylength = findMaxNodesInLayer();
      double[][][] weights = new
              double[connectivitylayers][weightarraylength][weightarraylength];

      try
      {

         BufferedReader reader = new BufferedReader(new FileReader(weightsfilepath));
         String line = reader.readLine();
         line = reader.readLine();

         while (line != null)
         {

            String[] weightindices = line.split(" ");
            int m = Integer.parseInt(weightindices[1]);
            int index1 = Integer.parseInt(weightindices[2]);
            int index2 = Integer.parseInt(weightindices[3]);
            double weightvalue = Double.parseDouble(weightindices[4]);

            weights[m][index1][index2] = weightvalue;
            line = reader.readLine();

         } //while (line != null)

      }
      catch (IOException e)
      {
         e.printStackTrace();
      }

      return weights;
   }

   /**
    * Generates a random weight within the specified range
    * @param lowerlim The lower limit for the random value
    * @param upperlim The upper limit for the random value
    * @return a generated random value within the specified range that can be utilized as an
    * initial weight for the network
    */
   public double generateRandomWeight(double lowerlim, double upperlim)
   {
      return Math.random() * (upperlim - lowerlim) + lowerlim;
   }
   
   /**
    * Loads the inputs for the given testcase into the input layer of the activations array.
    * @param testcase The current test case for which the inputs should be loaded into the input
    *                 layer of the activations array.
    */
   public void loadInputs(int testcase)
   {

      for (int i = 0; i < testsets[testcase][0].length; i++)
      {
         activations[0][i] = testsets[testcase][0][i];
         thetas[0][i] = testsets[testcase][0][i];
      }
   }

   /**
    * Creates a weights array full of random weight values by first initializing the weights
    * array and then setting each appropriate weight in the weights array to be a random value
    * within a user-specified range. Note that when the program loops through the weights array,
    * only weights that have indexes appropriate to the current network are set to random values;
    * all other weight values are set to zero. For example, if there is a 2-3-1 network, the
    * weight w[0][2][0] would be set to zero because the 3rd node in the input layer does not
    * technically exist; however, weight w[0][1][0] would be set to a random value because the
    * 2nd node in the input layer and 1st node in the hidden layer are both prevalent in the
    * network.
    * @return an array with random weights filled in
    */
   public double[][][] makeRandomWeightsArray()
   {
      int connectivitylayers = hiddenLayerNodes.length + 1;
      int weightarraylength = findMaxNodesInLayer();
      double[][][] weights = new double[connectivitylayers][weightarraylength][weightarraylength];


      Scanner scanner = new Scanner(System.in);
      System.out.println("Enter the lower limit for the weights");

      double lowerlim = scanner.nextDouble();
      System.out.println("Enter the upper limit for the weights");
      double upperlim = scanner.nextDouble();



      for (int m = 0; m < connectivitylayers; m++)
      {

         for (int indexlayer1 = 0; indexlayer1 < weightarraylength; indexlayer1++)
         {

            if (m==0&&indexlayer1<inputnodes ||
                    m==1&&indexlayer1<hiddenLayerNodes[0])
            {

               for (int indexlayer2 = 0; indexlayer2 < weightarraylength; indexlayer2++)
               {

                  if (m==1&&indexlayer2<outputnodes ||
                          m==0&&indexlayer2<hiddenLayerNodes[0])
                  {
                     weights[m][indexlayer1][indexlayer2] = generateRandomWeight(lowerlim,upperlim);
                  }

               } //for (int indexlayer2 = 0; indexlayer2 < weightarraylength; indexlayer2++)

            }

         } //for (int indexlayer1 = 0; indexlayer1 < weightarraylength; indexlayer1++)

      } //for (int m = 0; m < connectivitylayers; m++)

      return weights;
   }

   /**
    * Helper function that finds and returns the maximum number of nodes in all of the layers in
    * the network. For instance, if there is a 2-3-1 network, the maximum number of nodes in a
    * layer of the network would be 3. The output of this helper function is utilized in
    * determining the lengths of the weights and activations arrays.
    * @return The maximum number of nodes in a layer of the network
    */
   public int findMaxNodesInLayer()
   {
      int max = inputnodes;

      for (int i = 0; i<hiddenLayerNodes.length; i++)
      {

         if (hiddenLayerNodes[i]>max)
         {
            max = hiddenLayerNodes[i];
         }
      }

      if (max<outputnodes)
      {
         max = outputnodes;
      }

      return max;
   }

   /**
    * Calculates the error for the given test case by comparing the truth values from the test
    * sets and the calculated outputs.
    * @param testcase the test case for which the error should be calculated
    * @return the error for the given test case
    */
   public double calculateError(int testcase)
   {
      double errorval = 0.0;
      double[] calculatedOutputs = getOutputVals();

      for (int i = 0; i < calculatedOutputs.length; i++)
      {
         double outputdiff = testsets[testcase][1][i] - calculatedOutputs[i];
         errorval+=outputdiff*outputdiff;
      }

      return .5*errorval;
   }

   /**
    * Runs the network by filling in the values for all activations one layer at a time. Each
    * node's activation value is calculated by taking the dot product of the array representing
    * the values of the nodes in the previous layer and the array representing the weights of
    * the connections between the nodes of the previous layer and the current node to be
    * calculated. The activation function (sigmoid function in this case) is then applied to the dot
    * product sum, and the derived value is stored in the appropriate location of the activations
    * array. For the purposes of backpropagation, the raw value of the dot product sum (before
    * the activation function) is stored in the thetas array.
    */
   public void runNetwork()
   {
      for (int n=1; n<activations.length; n++)
      {

         for (int jki = 0; jki < activations[n].length; jki++)
         {
            double dotProductSum = 0.0;

            for (int index1 = 0; index1 < weights[n-1].length; index1++)
            {
               dotProductSum += activations[n-1][index1] * weights[n-1][index1][jki];
            }
            activations[n][jki] = activationFunction(dotProductSum);
            thetas[n][jki] = dotProductSum;
         }
      } //for (int n=1; n<activations.length; n++)
   }

   /**
    * Runs the neural network using the weights in the given file. First, the weights array is
    * created using the weights in the file. Then, the network is run for each test case, and the
    * output values and errors are printed to the console for each test case. The output values
    * are also stored in the user-specified file.
    * @param weightsfilepath the path of the file containing the weight values
    */
   public void runMode(String weightsfilepath)
   {
      weights = makeWeightsArray(weightsfilepath);
      Scanner scanner = new Scanner(System.in);

      System.out.println("Please enter the path for the file where you would like the outputs " +
              "saved");
      String outputfilepath = scanner.next();

      if (outputfilepath.equals("."))
      {
         outputfilepath = "outputs.txt";
      }

      try
      {
         BufferedWriter writer = new BufferedWriter(new FileWriter(outputfilepath));

         for (int testcase = 0; testcase < testsets.length; testcase++)
         {
            loadInputs(testcase);
            runNetwork();

            double error = calculateError(testcase);
            System.out.println("Output Vals for testcase " + (testcase + 1) + " are ");

            for (int i = 0; i < outputnodes; i++)
            {
               System.out.print(activations[activations.length-1][i] + ", ");
               writer.write(activations[activations.length-1][i] + " ");
            }
            System.out.println();
            System.out.println("Error for testcase " + (testcase + 1) + " is " + error);
            writer.newLine();
         } // for (int testcase = 0; testcase < testsets.length; testcase++)

         writer.close();
      }

      catch (IOException e)
      {
         e.printStackTrace();
      }
   }
   
   /**
    * Returns the values of the output nodes after the network has been run.
    * @return the values of the output nodes in the activations array.
    */
   public double[] getOutputVals()
   {
      double[] outputlayer = activations[activations.length-1];
      double[] outputvals = new double[outputnodes];

      for (int i = 0; i < outputnodes; i++)
      {
         outputvals[i] = outputlayer[i];
      }

      return outputvals;
   }
   
   /**
    * Calculates and applies the change in weights as described in the design document "4-Three
    * Plus Layer Network". Note that there are three steps (and therefore three loop
    * constructs) taken in calculating the change for all of the weights: the first is to
    * calculate how much the weights between the second hidden layer and the output layer should
    * change, the second is to calculate how much the weights between the first and second layer
    * should change, and the final is to calculate how much the weights between the input layer
    * and the hidden layer should change.
    * @param learningfactor the current learning factor
    * @param testcase the index of the testcase that the network is training for
    */
   public void backProp(double learningfactor, int testcase)
   {
      for (int i = 0; i < outputnodes; i++)
      {

         omegas[3][i] = testsets[testcase][1][i] - activations[3][i];
         psis[3][i] = omegas[3][i] * derivativeActivationFunction(thetas[3][i]);

         for (int j= 0; j < hiddenLayerNodes[1]; j++)
         {
            weights[2][j][i] += learningfactor * activations[2][j] * psis[3][i];
         }
      }

      for (int k = 0; k < hiddenLayerNodes[0] ; k++)
      {

         for (int j = 0; j < hiddenLayerNodes[1]; j++)
         {

            omegas[2][j] = 0.0;

            for (int i = 0; i<outputnodes; i++)
            {
               omegas[2][j] += psis[3][i] * weights[2][j][i];
            }

            psis[2][j] = omegas[2][j] * derivativeActivationFunction(thetas[2][j]);

            weights[1][k][j] += learningfactor * activations[1][k] * psis[2][j];
         } //for (int j = 0; j < hiddenLayerNodes[1]; j++)
      } //for (int k = 0; k < hiddenLayerNodes[0] ; k++)

      for (int m = 0; m < inputnodes; m++)
      {

         for (int k = 0; k < hiddenLayerNodes[0]; k++)
         {

            omegas[1][k] = 0.0;

            for (int j = 0; j < hiddenLayerNodes[1]; j++)
            {
               omegas[1][k] += psis[2][j] * weights[1][k][j];
            }

            psis[1][k] = omegas[1][k] * derivativeActivationFunction(thetas[1][k]);

            weights[0][m][k] += learningfactor * activations[0][m] * psis[1][k];
         } //for (int k = 0; k < hiddenLayerNodes[0]; k++)
      } //for (int m = 0; m < inputnodes; m++)
   }
   /**
    * Trains the network by running through each test case one at a time until one of the end
    * conditions are met:
    * End case 1: Maximum number of iterations are met
    * End case 2: Learning factor is less than learning factor threshold
    * End case 3: Error for each test case is less than the error threshold
    * The network trains using 6 different training parameters, specifically the learning factor,
    * the learning factor multiplier, the maximum number of iterations, the error threshold (the
    * minimum value the error must reach for each test case), the learning factor threshold (the
    * minimum value the learning factor can reach before the training is stopped), and printiter,
    * which indicates that debug statements should be printed to the console when the current
    * iteration number is divisible by the specified value.
    *
    * Training for the network occurs as follows:
    * 1) The weights array is initialized. If the user wants to randomize the weights, the
    * weights array is filled in with random values. Otherwise, the weights array is filled in
    * with values according to the given file.
    * 2) As long as none of the end conditions are met, a loop is run through the test cases. For
    * each test case, the inputs are loaded into the activations array, and the network is run.
    * 3) The delta weights are calculated using the partial derivative of the error with respect
    * to each weight and the learning factor. (backpropagation is applied to make this part of
    * the process significantly faster.) Note that the delta weights are both calculated annd
    * applied to the current weights within the backProp() method
    * 4) If the error decreases from the last iteration (for that test case), then the
    * learning factor is multiplied by the learning factor multiplier, and the error is saved.
    * However, if the error increases from the last iteration, the learning factor is divided
    * by the learning factor multiplier.
    * 5) After each iteration, a check is performed to see whether training has completed.
    * 6) After training is completed, the final weights are written to the file.
    *
    * @param weightsfilepath the path to the file containing the weights (if user chooses to use
    *                        non-random weights when training)
    * @param randomizeWeights a boolean representing whether or not to use random weights when
    *                         training
    */
   public void trainNetwork(boolean randomizeWeights, String weightsfilepath)
   {
      double learningfactor = trainingparams[0];
      double learningfactormultiplier = trainingparams[1];
      int maxiterations = (int) trainingparams[2];
      double errorthreshold = trainingparams[3];
      double learningfactorthreshold = trainingparams[4];
      int printiter = (int) trainingparams[5];


      if (randomizeWeights)
      {
         weights = makeRandomWeightsArray();
      }
      else
      {
         weights = makeWeightsArray(weightsfilepath);
      }

      double done = 0.0;
      int iteration = 1;
      double[] errors = new double[testsets.length];

      for (int i = 0; i < errors.length; i++)
      {
         errors[i] = Double.MAX_VALUE;
      }

      loadInputs(0);
      runNetwork();
      errors[0] = calculateError(0);

      while (done == 0)
      {

         for (int testcase = 0; testcase < testsets.length; testcase++)
         {

            loadInputs(testcase);
            runNetwork();

            boolean print = (iteration % printiter)<testsets.length;

            backProp(learningfactor, testcase);

            iteration++;

            double currenterror = calculateError(testcase);

            if (currenterror < errors[testcase])
            {
               learningfactor *= learningfactormultiplier;
               errors[testcase] = currenterror;
            }

            else if (currenterror > errors[testcase])
            {

               learningfactor /= learningfactormultiplier;
               errors[testcase] = currenterror;

            }

            if (iteration>maxiterations)
            {
               done = 1;
            }

            if (learningfactor < learningfactorthreshold)
            {
               done = 2;
            }

            boolean lessthanerrorthreshold = true;

            for (int tcase = 0; tcase < testsets.length; tcase++)
            {

               if (errors[tcase] >= errorthreshold)
               {
                  lessthanerrorthreshold = false;
               }
            }

            if (lessthanerrorthreshold)
            {
               done = 3;
            }

            if (print)
            {
               System.out.println("Current learning Factor after training iteration " +  (iteration-
                       1) + " " + learningfactor +  " and testcase number " + (testcase + 1));

               System.out.println("Current Output Vals after training iteration " + (iteration - 1)
                       + " and testcase number " + (testcase + 1) + " are ");

               for (int i = 0; i < outputnodes; i++)
               {
                  System.out.print(activations[activations.length-1][i] + ", ");
               }

               System.out.println();
               System.out.println("Current Error Val after training iteration " + (iteration - 1)  +
                       " " + errors[testcase]+ " and testcase number " + (testcase + 1));
            }

         } //for (int testcase = 0; testcase < testsets.length; testcase++)

      } // while (done == 0)
      
      double maxerror = 0.0;
      int maxtestcase = 0;

      for (int i = 0; i < errors.length; i++)
      {
         if (errors[i] > maxerror)
         {
            maxerror = errors[i];
            maxtestcase = i + 1;
         }
      }

      if (done == 1)
      {

         System.out.println("Stopped due to number of iterations exceeding " + maxiterations);
         System.out.println("Maximum error is " + maxerror + " and occurs at testcase " +
                 maxtestcase);

      }
      else if (done == 2)
      {
         System.out.println("Stopped due to learning factor being below learning factor " +
                 "threshold. Learning Factor is " + learningfactor);
      }

      else if (done == 3)
      {

         System.out.println("Stopped due to error of each test case being below error threshold. ");

         System.out.println("Maximum error is " + maxerror + " and occurs at testcase " +
                 maxtestcase);
         System.out.println("Number of iterations: " + iteration);

      }
      writeWeightsToFile();
   }
   
   /**
    * Method representing the activation function to be applied to the dot product sum
    * calculated for each node. Examples of this activation function include the sigmoid
    * function and the gaussian function. In this case, the activation function is the sigmoid
    * function.
    * @param dotproductsum The dot product sum calculated for the current node
    * @return The value after the activation function is applied to the dot product sum. In this
    * case, the output value of the sigmoid function on the given input is returned.
    */
   public double activationFunction(double dotproductsum)
   {
      return 1.0/(1.0 + Math.exp(-dotproductsum));
   }

   /**
    * Returns the derivative of the activation function, specifically the sigmoid function
    * @param dotproductsum the input value at which the derivative of the activation function
    *                      should be calculated
    * @return the value of the derivative of the activation function at the specified input value
    */
   public double derivativeActivationFunction(double dotproductsum)
   {
      double activationfuncval = activationFunction(dotproductsum);
      return activationfuncval * (1.0 - activationfuncval);
   }
   
   /**
    * Writes the weights to the file by iterating through the weights array and writing each one
    * to the weights file. Each weight is written on a new line in the following format: w (m)
    * (index of node in the first layer, i.e. input layer or hidden layer) (index of node in the
    * second layer, i.e. hidden layer or output layer)
    */
   public void writeWeightsToFile()
   {
      int connectivitylayers = weights.length;
      int weightarraylength = weights[0].length;

      try
      {

    	 BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/20arjuna/eclipse-workspace-oxygen/NeuralNetsP4/src/weights.txt"));
         writer.write("General format of weights file: w (m) (node index first layer) (node " +
                 "index second layer) (weight)");
         writer.newLine();

         for (int m = 0; m < connectivitylayers; m++)
         {

            for (int indexlayer1 = 0; indexlayer1 < weightarraylength; indexlayer1++)
            {

               for (int indexlayer2 = 0; indexlayer2 < weightarraylength; indexlayer2++)
               {

                  writer.write("w " + m + " " + indexlayer1 + " " + indexlayer2 + " " +
                          weights[m][indexlayer1][indexlayer2]);
                  writer.newLine();

               }
            } //for (int indexlayer1 = 0; indexlayer1 < weightarraylength; indexlayer1++)
         } //for (int m = 0; m < connectivitylayers; m++)

         writer.close();

      }

      catch(IOException e)
      {
         e.printStackTrace();
      }
   }
}
