package HomeWork4;

import java.util.Arrays;

import weka.classifiers.Classifier;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class InstanceNeighbor implements Comparable<InstanceNeighbor> {
	private Instance instance;
	private double distance;

	public InstanceNeighbor(Instance instance, double distance) {
		setDistance(distance);
		setInstance(instance);
	}

	public Instance getInstance() {
		return instance;
	}

	public double getDistance() {
		return distance;
	}

	public void setInstance(Instance instance) {
		this.instance = instance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	@Override
	public int compareTo(InstanceNeighbor otherInstance) {
		if (otherInstance.getDistance() > this.getDistance()) {
			return -1;
		} else if (otherInstance.getDistance() < this.getDistance()) {
			return 1;
		}
		return 0;
	}
}

public class Knn implements Classifier {

	public enum EditMode {
		None, Forwards, Backwards
	};

	private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;
	private int m_totalNumberOfTrainingInstances = 0;
	private Instances[] m_folded;
	private int m_bestK;

	public enum PhaseMode {
		Phase1, Phase2
	};

	private PhaseMode m_phaseMode = PhaseMode.Phase1;

	private long[] m_oneFoldTime;
	private long m_totalFoldTime;
	private double m_avgFoldTime;

	public enum lpDistanceMode {
		Infinity, One, Two, Three
	};

	private lpDistanceMode m_distanceMode;

	public enum MajorityMode {
		Uniform, Weighted
	};

	private MajorityMode m_MajorityMode;

	public EditMode getEditMode() {
		return m_editMode;
	}

	public void setEditMode(EditMode editMode) {
		m_editMode = editMode;
	}

	public int getBestK() {
		return m_bestK;
	}

	public lpDistanceMode getDistanceMode() {
		return m_distanceMode;
	}

	public MajorityMode getMajorityMode() {
		return m_MajorityMode;
	}

	public void setM_editMode(EditMode m_editMode) {
		this.m_editMode = m_editMode;
	}

	public void setM_phaseMode(PhaseMode m_phaseMode) {
		this.m_phaseMode = m_phaseMode;
	}

	public double getM_avgFoldTime() {
		return m_avgFoldTime;
	}

	public long getM_totalFoldTime() {
		return m_totalFoldTime;
	}

	public Instances getM_trainingInstances() {
		return m_trainingInstances;
	}

	public void setM_folded(Instances[] m_folded) {
		this.m_folded = m_folded;
	}

	public int getM_totalNumberOfTrainingInstances() {
		return m_totalNumberOfTrainingInstances;
	}

	public void initKnn(int bestK, lpDistanceMode bestDistanceMode, MajorityMode bestMajorityMode, Instances data) {
		m_bestK = bestK;
		m_distanceMode = bestDistanceMode;
		m_MajorityMode = bestMajorityMode;
		m_trainingInstances = new Instances(data);
	}

	/**
	 * Finds the best hyper parameters (K number of neighbors, l-p distance
	 * measure, majority method) using 10-folds cross validation.
	 */
	public void findBestHyperParameters() {
		double bestParametersAvgError = Double.MAX_VALUE;
		int bestK = 0;
		lpDistanceMode bestDistanceMode = null;
		MajorityMode bestMajorityMode = null;

		setM_folded(foldInstances(10));
		lpDistanceMode[] distanceModes = { lpDistanceMode.Infinity, lpDistanceMode.One, lpDistanceMode.Two,
				lpDistanceMode.Three };
		MajorityMode[] majorityModes = { MajorityMode.Uniform, MajorityMode.Weighted };

		for (int k = 1; k <= 20; k++) {
			m_bestK = k;
			for (lpDistanceMode distanceMode : distanceModes) {
				m_distanceMode = distanceMode;
				for (MajorityMode majorityMode : majorityModes) {
					m_MajorityMode = majorityMode;
					double currentParamatersAvgError = crossValidationError();
					if (currentParamatersAvgError < bestParametersAvgError) {
						bestParametersAvgError = currentParamatersAvgError;
						bestK = k;
						bestDistanceMode = distanceMode;
						bestMajorityMode = majorityMode;
					}
				}
			}
		}
		m_bestK = bestK;
		m_distanceMode = bestDistanceMode;
		m_MajorityMode = bestMajorityMode;
	}

	/**
	 * Calculate the cross validation error: average error on all folds.
	 * 
	 * @return average fold error.
	 */
	public double crossValidationError() {
		double crossValidationError = 0;
		m_totalNumberOfTrainingInstances = 0;

		for (int i = 0; i < m_folded.length; i++) {
			Instances validationInstances = new Instances(m_folded[i]);
			for (int j = 0; j < m_folded[i].size(); j++) {
				m_trainingInstances.delete(0);
			}
			Instances currentTrainingInstances = new Instances(m_trainingInstances);
			if (m_phaseMode == PhaseMode.Phase2) {
				try {
					buildClassifier(m_trainingInstances);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			m_totalNumberOfTrainingInstances += m_trainingInstances.size();
			long startTime = System.nanoTime();
			double currentValidationAvgError = calcAvgError(validationInstances);
			long stopTime = System.nanoTime();
			m_oneFoldTime[i] = stopTime - startTime;
			crossValidationError += currentValidationAvgError;
			m_trainingInstances = currentTrainingInstances;
			for (Instance instance : validationInstances) {
				m_trainingInstances.add(instance);
			}

		}
		if (m_phaseMode == PhaseMode.Phase2) {
			m_totalFoldTime = 0;
			for (int i = 0; i < m_oneFoldTime.length; i++) {
				m_totalFoldTime += m_oneFoldTime[i];
			}
			m_avgFoldTime = m_totalFoldTime / (double) m_oneFoldTime.length;
		}
		return crossValidationError /= (double) m_folded.length;
	}

	/**
	 * Splits the data into 'numberOfFolds' folds.
	 * 
	 * @param numberOfFolds
	 *            - number of folds to split the data into.
	 * @return the data split into 'numberOfFolds' folds.
	 */
	public Instances[] foldInstances(int numberOfFolds) {
		m_oneFoldTime = new long[numberOfFolds];
		Instances[] foldedInstances = new Instances[numberOfFolds];
		int sizeOfFold = (m_trainingInstances.size() / numberOfFolds);
		int j = 0;
		for (int i = 0; i < foldedInstances.length; i++, j += sizeOfFold) {
			foldedInstances[i] = new Instances(m_trainingInstances, j, sizeOfFold);
		}
		int remaningInstances = m_trainingInstances.size() - j;
		for (int i = 0; i < remaningInstances; i++) {
			foldedInstances[i].add(m_trainingInstances.instance(i + j));
		}
		return foldedInstances;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (m_editMode) {
		case None:
			noEdit(arg0);
			break;
		case Forwards:
			editedForward(arg0);
			break;
		case Backwards:
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}

	@Override
	public double classifyInstance(Instance instance) {
		InstanceNeighbor[] nearestNeighbors = findNearestNeighbors(instance);
		if (m_MajorityMode == MajorityMode.Uniform) {
			return getClassVoteResult(nearestNeighbors);
		} else {
			return getWeightedClassVoteResult(nearestNeighbors);
		}
	}

	/**
	 * Calculates the average error on a given instances set. The average error
	 * is the total number of classification mistakes on the input instances
	 * set, divided by the number of instances in the input set.
	 * 
	 * @param data
	 *            - the given instances set.
	 * @return average error
	 */
	public double calcAvgError(Instances data) {
		int numberOfMistakes = 0;
		for (Instance instance : data) {
			if (classifyInstance(instance) != instance.classValue()) {
				numberOfMistakes++;
			}
		}
		return (double) numberOfMistakes / (double) data.size();
	}

	/**
	 * Calculates the Precision & Recall on a given instances set.
	 * 
	 * @param data
	 *            - the given instances set.
	 * @return double array of size 2. First index for Precision and the second
	 *         for Recall.
	 */
	public double[] calcConfusion(Instances data) {
		int numberOfTruePositives = 0, numberOfFalsePositives = 0, numberOfFalseNegatives = 0;
		double[] calcConfusion = new double[2];
		for (Instance instance : data) {
			double prediction = classifyInstance(instance);
			if ((prediction == instance.classValue()) && (prediction == 0)) {
				numberOfTruePositives++;
			} else if ((prediction != instance.classValue()) && (prediction == 0)) {
				numberOfFalsePositives++;
			} else if ((prediction != instance.classValue()) && (prediction == 1)) {
				numberOfFalseNegatives++;
			}
		}
		calcConfusion[0] = (double) numberOfTruePositives / (double) (numberOfTruePositives + numberOfFalsePositives);
		calcConfusion[1] = (double) numberOfTruePositives / (double) (numberOfTruePositives + numberOfFalseNegatives);
		return calcConfusion;
	}

	/**
	 * Calculates the average confusion on all folds.
	 * 
	 * @return double array of size 2. First index for average Precision and the
	 *         second for average Recall.
	 */
	public double[] avgCalcConfusion() {
		double[] avgCalcConfusion = new double[2];
		for (int i = 0; i < m_folded.length; i++) {
			Instances validationInstances = new Instances(m_folded[i]);
			for (int j = 0; j < m_folded[i].size(); j++) {
				m_trainingInstances.delete(0);
			}
			double[] currentCalcConfusion = calcConfusion(validationInstances);
			avgCalcConfusion[0] += currentCalcConfusion[0];
			avgCalcConfusion[1] += currentCalcConfusion[1];
			for (Instance instance : validationInstances) {
				m_trainingInstances.add(instance);
			}
		}
		avgCalcConfusion[0] /= (double) m_folded.length;
		avgCalcConfusion[1] /= (double) m_folded.length;
		return avgCalcConfusion;
	}

	/**
	 * Finds the K nearest neighbors for the instance being classified.
	 * 
	 * @param instance
	 *            - an instance that is being classified.
	 * @return the K nearest neighbors and their distances.
	 */
	public InstanceNeighbor[] findNearestNeighbors(Instance instance) {
		int i = 0;
		InstanceNeighbor[] allNeighbors = new InstanceNeighbor[m_trainingInstances.size()];
		InstanceNeighbor[] nearestNeighbors;

		if (m_trainingInstances.size() < m_bestK) {
			nearestNeighbors = new InstanceNeighbor[m_trainingInstances.size()];
		} else {
			nearestNeighbors = new InstanceNeighbor[m_bestK];
		}

		for (Instance currentInstance : m_trainingInstances) {
			allNeighbors[i] = new InstanceNeighbor(currentInstance, distance(instance, currentInstance));
			i++;
		}
		Arrays.sort(allNeighbors);
		System.arraycopy(allNeighbors, 0, nearestNeighbors, 0, nearestNeighbors.length);
		return nearestNeighbors;
	}

	/**
	 * Calculates the majority class of the neighbors.
	 * 
	 * @param nearestNeighbors
	 *            - a set of K nearest neighbors.
	 * @return the majority vote on the class of the neighbors.
	 */
	public double getClassVoteResult(InstanceNeighbor[] nearestNeighbors) {
		int[] classVoteResult = new int[m_trainingInstances.classAttribute().numValues()];
		for (InstanceNeighbor instanceNeighbor : nearestNeighbors) {
			classVoteResult[(int) instanceNeighbor.getInstance().classValue()]++;
		}
		int maxVotes = classVoteResult[0];
		int indexOfMaxVotes = 0;
		for (int i = 1; i < classVoteResult.length; i++) {
			if (classVoteResult[i] > maxVotes) {
				maxVotes = classVoteResult[i];
				indexOfMaxVotes = i;
			}
		}
		return indexOfMaxVotes;
	}

	/**
	 * Calculates the weighted majority class of the neighbors. In this method
	 * the class vote is normalized by the distance from the instance being
	 * classified.
	 * 
	 * @param nearestNeighbors
	 *            - a set of K nearest neighbors and their distances.
	 * @return the majority vote on the class of the neighbors, where each
	 *         neighbor's class is weighted by the neighbor’s distance from the
	 *         instance being classified.
	 */
	public double getWeightedClassVoteResult(InstanceNeighbor[] nearestNeighbors) {
		double[] classVoteResult = new double[m_trainingInstances.classAttribute().numValues()];
		for (InstanceNeighbor instanceNeighbor : nearestNeighbors) {
			classVoteResult[(int) instanceNeighbor.getInstance().classValue()] += (1.0
					/ Math.pow(instanceNeighbor.getDistance(), 2));
		}
		double maxVotes = classVoteResult[0];
		int indexOfMaxVotes = 0;
		for (int i = 1; i < classVoteResult.length; i++) {
			if (classVoteResult[i] > maxVotes) {
				maxVotes = classVoteResult[i];
				indexOfMaxVotes = i;
			}
		}
		return indexOfMaxVotes;
	}

	/**
	 * Calculates 2 input instances’ distance according to the distance function
	 * that the current algorithm is configured to use.
	 * 
	 * @param instance1
	 * @param instance2
	 * @return the input instances’ distance according to the distance function
	 *         that the current algorithm is configured to use.
	 */
	public double distance(Instance instance1, Instance instance2) {
		if (m_distanceMode == lpDistanceMode.Infinity) {
			return lInfinityDistance(instance1, instance2);
		}
		return lPDistance(instance1, instance2);
	}

	/**
	 * Calculates the l-p distance between two instances, according to the p
	 * parameter that the current algorithm is configured to use.
	 * 
	 * @param instance1
	 * @param instance2
	 * @return the l-p distance between the two instances.
	 */
	public double lPDistance(Instance instance1, Instance instance2) {
		int pParameter = 0;
		switch (m_distanceMode) {
		case One:
			pParameter = 1;
			break;
		case Two:
			pParameter = 2;
			break;
		default:
			pParameter = 3;
			break;
		}
		double distance = 0;
		for (int d = 0; d < instance1.numAttributes(); d++) {
			if ((d != instance1.classIndex()) && !(instance1.attribute(d).name().equals("id"))) {
				distance += Math.abs(Math.pow((instance1.value(d) - instance2.value(d)), pParameter));
			}
		}
		return Math.pow(distance, ((double) 1 / (double) pParameter));
	}

	/**
	 * Calculates the l-infinity distance between two instances.
	 * 
	 * @param instance1
	 * @param instance2
	 * @return the l-infinity distance between the two instances.
	 */
	public double lInfinityDistance(Instance instance1, Instance instance2) {
		double maxDistance = 0;
		for (int d = 0; d < instance1.numAttributes(); d++) {
			if ((d != instance1.classIndex()) && !(instance1.attribute(d).name().equals("id"))) {
				double currentDistance = Math.abs(instance1.value(d) - instance2.value(d));
				if (currentDistance > maxDistance) {
					maxDistance = currentDistance;
				}
			}
		}
		return maxDistance;
	}

	/**
	 * Stores the training set in the m_trainingInstances using the forwards
	 * editing.
	 * 
	 * @param instances - training instances to edit.
	 */
	private void editedForward(Instances instances) {
		m_trainingInstances = new Instances(instances, 0, 1);
		for (Instance instance : instances) {
			if (classifyInstance(instance) != instance.classValue()) {
				m_trainingInstances.add(instance);
			}
		}
	}

	/** Stores the training set in the m_trainingInstances using the backwards
	 * editing.
	 * 
	 * @param instances - training instances to edit.
	 */
	private void editedBackward(Instances instances) {
		m_trainingInstances = new Instances(instances);
		for (Instance instance : instances) {
			m_trainingInstances.delete(0);
			if (classifyInstance(instance) != instance.classValue()) {
				m_trainingInstances.add(instance);
			}
		}
	}

	/**
	 * Stores the training set in the m_trainingInstances without editing.
	 * 
	 * @param instances - training instances.
	 */
	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
}
