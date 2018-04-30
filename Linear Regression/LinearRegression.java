/**
 * @author Tomer Korbinsky & Arad Zaltsberger
 * ID's: 203021720, 312533342
 */
package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	// the method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		// since class attribute is also an attribute we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;
		setAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);

	}

	/**
	 * sets alpha by the recitations algorithm
	 * 
	 * @param data
	 *            - the learning data
	 * @throws Exception
	 */
	private void setAlpha(Instances data) throws Exception {
		// keeping record of the min alpha value
		double minAlpha = Math.pow(3, -17);
		m_alpha = Math.pow(3, -17);
		double minSE = Double.MAX_VALUE;
		// iterate through alpha options from 3^-17 to 3^2
		while (m_alpha <= (9)) {
			// initialize coefficients by guess to 1
			initCoeffs();
			for (int i = 0; i < 20000; i++) {
				updateCoeffs(data);
			}
			double currentSE = calculateSE(data);
			if (currentSE < minSE) {
				minAlpha = m_alpha;
				minSE = currentSE;
			}
			m_alpha *= 3;
		}
		// sets the "best alpha"
		m_alpha = minAlpha;
	}

	/**
	 * An implementation of the gradient descent algorithm which should return
	 * the weights of a linear regression predictor which minimizes the average
	 * squared error.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		// initializing "artificial" coefficients
		initCoeffs();
		// calculate SE for current coefficients
		double currentSE = calculateSE(trainingData);
		// compare the current SE to the SE we calculated 100 iterations ago
		double SEDiffrence = Double.MAX_VALUE;
		// counter to check SEDiffrence each 100 iterations
		int counter = 1;
		// update coefficients until SE is decreased by 0.003 after 100
		// iterations
		while (SEDiffrence > 0.003) {
			updateCoeffs(trainingData);
			if (counter == 100) {
				SEDiffrence = currentSE - calculateSE(trainingData);
				currentSE = calculateSE(trainingData);
				counter = 0;
			}
			counter++;
		}
		return m_coefficients;
	}

	/**
	 * initialize coefficients by guess to 1
	 */
	private void initCoeffs() {
		m_coefficients = new double[m_truNumAttributes + 1];
		for (int i = 0; i < m_coefficients.length; i++) {
			m_coefficients[i] = 1;
		}
	}

	/**
	 * updates coefficient according to gradient descent algorithm.
	 * 
	 * @param trainingData
	 */
	private void updateCoeffs(Instances trainingData) {
		// create a temporary array of coefficients to make simultaneous updates
		double[] tempCoeffs = new double[m_coefficients.length];
		for (int i = 0; i < tempCoeffs.length; i++) {
			tempCoeffs[i] = m_coefficients[i] - (m_alpha * partialSEDerivative(trainingData, i));

		}
		// update the coefficient according to the temporary coefficient
		m_coefficients = tempCoeffs.clone();
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
	 *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		// the prediction is the result of the inner product of the coefficients
		// with the instance's attribute
		double prediction = m_coefficients[0];
		for (int i = 0; i < instance.numAttributes(); i++) {
			if (i != m_ClassIndex) {
				prediction += (m_coefficients[i + 1] * instance.value(i));
			}
		}
		return prediction;
	}

	/**
	 * function to calculate partial derivative of the SE for a specific
	 * coefficient
	 * 
	 * @param trainingData
	 *            - the learning data
	 * @param currentCoefIndex
	 *            - the index of coefficient, the partial derivative is in
	 *            respect to.
	 * @return - partial derivative of the SE in respect to currentCoefIndex
	 */
	private double partialSEDerivative(Instances trainingData, int currentCoefIndex) {
		double partialDerivative = 0;
		// iterate through all instances to calculate the sigma of formula
		for (int d = 0; d < trainingData.numInstances(); d++) {
			Instance myInstance = trainingData.instance(d);
			double prediction = m_coefficients[0];
			// iterate through all coefficients besides the first one
			for (int i = 0; i < trainingData.numAttributes(); i++) {
				if (i != m_ClassIndex) {
					prediction += (m_coefficients[i + 1] * myInstance.value(i));
				}
			}
			prediction -= myInstance.classValue();
			if (currentCoefIndex != 0) {
				prediction *= myInstance.value(currentCoefIndex - 1);
			}
			partialDerivative += prediction;
		}
		return (partialDerivative / trainingData.numInstances());
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
	 *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateSE(Instances data) throws Exception {
		double SE = 0;
		// iterate through all instances to calculate the sigma of formula
		for (int d = 0; d < data.numInstances(); d++) {
			Instance myInstance = data.instance(d);
			// calculate SE according to the formula
			double prediction = regressionPrediction(myInstance);
			prediction -= myInstance.classValue();
			prediction *= prediction;
			SE += prediction;
		}
		return (SE / (data.numInstances() * 2));
	}

	/**
	 * Prints the coefficients
	 * 
	 */
	public void printCoeffs() {
		for (int i = 0; i < m_coefficients.length; i++) {
			if (i != m_coefficients.length - 1) {
				System.out.print(m_coefficients[i] + ", ");
			} else {
				System.out.print(m_coefficients[i]);
			}
		}
		System.out.println();
	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
