package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import HomeWork4.Knn.EditMode;
import weka.core.Instances;

public class MainHW4 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		/* Phase 1 */

		/* Glass Data */
		Instances glassData = loadData("glass.txt");
		Random random1 = new Random();
		glassData.randomize(random1);
		Knn knnGlass = new Knn();
		knnGlass.buildClassifier(glassData);
		knnGlass.findBestHyperParameters();
		System.out.println("Cross validation error with K = " + knnGlass.getBestK() + ", p = "
				+ knnGlass.getDistanceMode() + ", majority function = " + knnGlass.getMajorityMode()
				+ " for glass data is: " + knnGlass.crossValidationError());

		/* Cancer Data */
		Instances cancerData = loadData("cancer.txt");
		Random random2 = new Random();
		cancerData.randomize(random2);
		Knn knnCancer = new Knn();
		knnCancer.buildClassifier(cancerData);
		knnCancer.findBestHyperParameters();
		System.out.println("Cross validation error with K = " + knnCancer.getBestK() + ", p = "
				+ knnCancer.getDistanceMode() + ", majority function = " + knnCancer.getMajorityMode()
				+ " for cancer data is: " + knnCancer.crossValidationError());
		double[] avgConfusionCancer = knnCancer.avgCalcConfusion();
		System.out.println("The average Precision for the cancer dataset is: " + avgConfusionCancer[0]);
		System.out.println("The average Recall for the cancer dataset is: " + avgConfusionCancer[1]);

		/* Phase 2 */
		int bestK = knnGlass.getBestK();
		Knn.lpDistanceMode lPDistanceMode = knnGlass.getDistanceMode();
		Knn.MajorityMode majorityMode = knnGlass.getMajorityMode();
		int[] foldSizes = { glassData.size(), 50, 10, 5, 3 };
		Knn.EditMode[] editMode = { EditMode.None, EditMode.Forwards, EditMode.Backwards };
		for (int i = 0; i < foldSizes.length; i++) {
			System.out.println("----------------------------");
			System.out.println("Results for " + foldSizes[i] + " folds");
			System.out.println("----------------------------");
			for (int j = 0; j < editMode.length; j++) {
				knnGlass = new Knn();
				knnGlass.initKnn(bestK, lPDistanceMode, majorityMode, glassData);
				knnGlass.setM_phaseMode(Knn.PhaseMode.Phase2);
				knnGlass.setEditMode(editMode[j]);
				knnGlass.setM_folded(knnGlass.foldInstances(foldSizes[i]));
				/* knnGlass.buildClassifier(glassData); */
				double error = knnGlass.crossValidationError();
				System.out.println("Cross validation error of " + editMode[j] + "-Edited knn on glass dataset is "
						+ error + " and the average elapsed time is " + knnGlass.getM_avgFoldTime());
				System.out.println("The total elapsed time is: " + knnGlass.getM_totalFoldTime());
				System.out.println("The total number of instances used in the classification phase is: "
						+ knnGlass.getM_totalNumberOfTrainingInstances());
			}
		}
	}
}
