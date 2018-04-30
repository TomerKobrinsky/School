/**
 * @author Tomer Korbinsky & Arad Zaltsberger
 * ID's: 203021720, 312533342
 */
package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * 
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		// load data
		Instances trainingData = loadData("wind_training.txt");
		// train classifier
		LinearRegression linearRegression = new LinearRegression();
		linearRegression.buildClassifier(trainingData);
		// calculate error on test data
		Instances testingData = loadData("wind_testing.txt");
		double SE = linearRegression.calculateSE(testingData);
		System.out.print("The weights are: ");
		linearRegression.printCoeffs();
		System.out.println("The error is: " + SE);
	}

}
