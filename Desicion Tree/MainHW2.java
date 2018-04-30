/**
 * @author Tomer Korbinsky & Arad Zaltsberger
 * ID's: 203021720, 312533342
 */
 
 package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork2.DecisionTree.PruningMode;
import weka.core.Instances;

public class MainHW2 {

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
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");

		double averageErrorTrain;
		double averageErrorTest;
		int numOfRules;
		for (int pruningMode = 0; pruningMode < 3; pruningMode++) {
			DecisionTree decisionTree = new DecisionTree();
			
			if (pruningMode == 0) {
				decisionTree.setPruningMode(DecisionTree.PruningMode.None);
				decisionTree.buildClassifier(trainingCancer);
				averageErrorTrain = decisionTree.calcAvgError(trainingCancer);
				averageErrorTest = decisionTree.calcAvgError(testingCancer);
				numOfRules = decisionTree.getNumOfRules();
				System.out.println("Decision Tree with No pruning \n"
						+ "The average train error of the decision tree is " + averageErrorTrain
						+ "\nThe average test error of the decision tree is " + averageErrorTest
						+ "\nThe amount of rules generated from the tree " + numOfRules);
			} else if (pruningMode == 1) {
				decisionTree.setPruningMode(DecisionTree.PruningMode.Chi);
				decisionTree.buildClassifier(trainingCancer);
				averageErrorTrain = decisionTree.calcAvgError(trainingCancer);
				averageErrorTest = decisionTree.calcAvgError(testingCancer);
				numOfRules = decisionTree.getNumOfRules();
				System.out.println("Decision Tree with Chi pruning \n"
						+ "The average train error of the decision tree with Chi pruning is " + averageErrorTrain
						+ "\nThe average test error of the decision tree Chi pruning is " + averageErrorTest
						+ "\nThe amount of rules generated from the tree " + numOfRules);
			} else {
				decisionTree.setPruningMode(DecisionTree.PruningMode.Rule);
				decisionTree.setValidation(validationCancer);
				decisionTree.buildClassifier(trainingCancer);
				averageErrorTrain = decisionTree.calcAvgError(trainingCancer);
				averageErrorTest = decisionTree.calcAvgError(testingCancer);
				numOfRules = decisionTree.getNumOfRules();
				System.out.println("Decision Tree with Rule pruning \n"
								+ "The average train error of the decision tree with Rule pruning is " + averageErrorTrain
								+ "\nThe average test error of the decision tree Rule pruning is " + averageErrorTest
								+ "\nThe amount of rules generated from the tree " + numOfRules);
			}
		}
	}
}
