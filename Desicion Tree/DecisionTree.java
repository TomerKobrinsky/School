/**
 * @author Tomer Korbinsky & Arad Zaltsberger
 * ID's: 203021720, 312533342
 */
 
package HomeWork2;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.IntStream;

import org.omg.CORBA.Current;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

class BasicRule {
	int attributeIndex;
	int attributeValue;

	/**
	 * constructor for basic rule
	 * 
	 * @param attributeIndex
	 * @param attributeValue
	 */
	public BasicRule(int attributeIndex, int attributeValue) {
		this.attributeIndex = attributeIndex;
		this.attributeValue = attributeValue;
	}
}

class Rule {
	List<BasicRule> basicRules;
	double returnValue = -1;
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue = -1;
	Rule nodeRule = new Rule();

}

public class DecisionTree implements Classifier {
	private Node rootNode;

	public enum PruningMode {
		None, Chi, Rule
	};

	private PruningMode m_pruningMode;
	Instances validationSet;
	private List<Rule> rules = new ArrayList<Rule>();

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		buildTree(arg0);
		convertTreetoRules(rootNode);
		if (m_pruningMode == PruningMode.Rule) {
			rulePruning();
		}
	}

	public double calcAvgError(Instances data) {
		int numberOfMistakes = 0;
		for (Instance instance : data) {
			if (classifyInstance(instance) != instance.classValue()) {
				numberOfMistakes++;
			}
		}
		return (double) numberOfMistakes / (double) data.size();
	}

	private double calcChiSquare(Instances data, int attributeIndex) {
		double chiSquare = 0;
		Instances splittedData[] = splitData(data, attributeIndex);
		double probabilities[] = getProbabilities(data);
		for (int i = 0; i < data.attribute(attributeIndex).numValues(); i++) {
			int pF = 0, nF = 0, dF = 0;
			if (splittedData[i] != null) {
				dF = splittedData[i].size();
				for (Instance instance : splittedData[i]) {
					if (instance.classValue() == 0) {
						pF++;
					} else {
						nF++;
					}
				}
				double e0 = probabilities[0] * dF;
				double e1 = probabilities[1] * dF;
				chiSquare += (Math.pow((pF - e0), 2) / e0);
				chiSquare += (Math.pow((nF - e1), 2) / e1);
			}
		}
		return chiSquare;
	}
	
	private void rulePruning() {
		if (rules.size() <= 1) {
			return;
		}
		List<Rule> pruningRules = new ArrayList<Rule>(rules);
		double avgErrorBefore, avgErrorAfter;
		double bestImprovement = 0;
		int indexToRemove = -1;
		for (int i = 0; i < rules.size(); i++) {
			avgErrorBefore = calcAvgError(validationSet);
			pruningRules.remove(i);
			List<Rule> temp = new ArrayList<Rule>(rules);
			rules = pruningRules;
			avgErrorAfter = calcAvgError(validationSet);
			if ((avgErrorAfter - avgErrorBefore) < bestImprovement) {
				bestImprovement = avgErrorAfter - avgErrorBefore;
				indexToRemove = i;
			}
			rules = temp;
			pruningRules = rules;
		}
		if (bestImprovement != 0) {
			rules.remove(indexToRemove);
			rulePruning();
		} else {
			return;
		}
	}

	/**
	 * converts decision tree to an arrayList of rules recursively
	 * 
	 * @param subRoot
	 *            - the root of the subTree
	 */
	private void convertTreetoRules(Node subRoot) {
		/*
		 * stopping condition. if subRoot has a non-negative return value it's a
		 * leaf Therefore, this leaf's rule is a rule of the decision tree
		 */
		if (subRoot.returnValue >= 0) {
			rules.add(subRoot.nodeRule);
			return;
		}
		// otherwise, apply the function on all subRoot'S children
		for (int i = 0; i < subRoot.children.length; i++) {
			if (subRoot.children[i] != null) {
				convertTreetoRules(subRoot.children[i]);
			}
		}
	}

	/**
	 * Builds the decision tree on a given data set using a recursive helper
	 * function
	 * 
	 * @param data
	 *            - the training data
	 */
	private void buildTree(Instances data) {
		// create the root node
		rootNode = new Node();
		rootNode.nodeRule.basicRules = new ArrayList<BasicRule>();
		rootNode.attributeIndex = findBestAttribute(data);
		// start building the tree recursively
		buildTreeRec(data, rootNode);
	}

	/**
	 * Builds the decision tree on a given data set recursively
	 * 
	 * @param data
	 *            - the training data or a subset of it
	 * @param subRoot
	 *            - the root of the subTree
	 */
	private void buildTreeRec(Instances data, Node subRoot) {
		
		/*
		 * stopping condition, if the data in this node identical,
		 * but there are different class values, we will enter this condition and make it a leaf
		 */
		if (subRoot.attributeIndex == -1) {
			subRoot.returnValue = findMajorityByInstances(data);
			subRoot.nodeRule.returnValue = subRoot.returnValue;
			return;
		}
		/*
		 * stopping condition, if the data is perfectly classified then the
		 * subRoot is a leaf and all instances in data have the same class
		 */
		if (calcEntropy(getProbabilities(data)) == 0) {
			// set the correct return value of subRoot
			subRoot.returnValue = data.instance(0).classValue();
			subRoot.nodeRule.returnValue = subRoot.returnValue;
			return;
		}

		/* check if chi pruning mode is set and check its condition */
		if (m_pruningMode == PruningMode.Chi && calcChiSquare(data, subRoot.attributeIndex) < 15.51) {
			subRoot.returnValue = findMajorityByInstances(data);
			subRoot.nodeRule.returnValue = subRoot.returnValue;
			return;
		}
		
		// split the given data according to subRoot's attribute values
		Instances childrenData[] = splitData(data, subRoot.attributeIndex);
		subRoot.children = new Node[childrenData.length];
		
		// create a child node for each value of subroot's attribute
		for (int i = 0; i < childrenData.length; i++) {
			Node node = new Node();
			node.parent = subRoot;
			node.nodeRule.basicRules = new ArrayList<BasicRule>();

			// add the rule of subRoot to it's child, and add the current
			// basic
			// rule
			if (!subRoot.nodeRule.basicRules.isEmpty()) {
				node.nodeRule.basicRules.addAll(subRoot.nodeRule.basicRules);
			}
			BasicRule basicRule = new BasicRule(subRoot.attributeIndex, i);
			node.nodeRule.basicRules.add(basicRule);
			
			// if there are instances in this child
			if (childrenData[i] != null) {
				
				node.attributeIndex = findBestAttribute(childrenData[i]);
				subRoot.children[i] = node;

				// build subTree with child as subRoot, and the correct subData
				buildTreeRec(childrenData[i], subRoot.children[i]);
			
			// if there are no instances in this child, make this child a leaf
			} else {
				node.attributeIndex = -1;
				node.returnValue = findMajorityByInstances(data);
				node.nodeRule.returnValue = node.returnValue;
				subRoot.children[i] = node;
				return;
			}
		}
	}

	/**
	 * finds the best decision attribute to split the next node according to the
	 * max gain ratio
	 * 
	 * @param data
	 *            - data in current node
	 * @return - the best attribute
	 */
	private int findBestAttribute(Instances data) {
		// if entropy is zero, there's no meaning for finding best attribute
		if (calcEntropy(getProbabilities(data)) == 0) {
			return -1;
		}
		int bestAttribute = -1;
		double bestGainRatio = -1;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			double currentGainRatio = gainRatio(data, i);
			if (currentGainRatio > bestGainRatio) {
				bestAttribute = i;
				bestGainRatio = currentGainRatio;
			}
		}
		return bestAttribute;
	}

	/**
	 * calculates the gain ratio according to the formula
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return - gain ratio
	 */
	private double gainRatio(Instances data, int attributeIndex) {
		return (calcInfoGain(data, attributeIndex) / splitInformation(data, attributeIndex));
	}

	/**
	 * calculates the information according the formula
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return - information gain
	 */
	private double calcInfoGain(Instances data, int attributeIndex) {
		double sum = 0;
		double entropyS = calcEntropy(getProbabilities(data));
		int numValues = data.attribute(attributeIndex).numValues();
		// split the given data according to attributeIndex values
		Instances splittedData[] = splitData(data, attributeIndex);

		for (int i = 0; i < numValues; i++) {
			if (splittedData[i] != null) {
				sum += ((double) splittedData[i].size() / (double) data.size())
						* calcEntropy(getProbabilities(splittedData[i]));
			}
		}

		return entropyS - sum;
	}

	/**
	 * calculates split information according to the formula
	 * 
	 * @param data
	 * @param attributeIndex
	 * @return
	 */
	private double splitInformation(Instances data, int attributeIndex) {
		double splitInformation = 0;
		int numValues = data.attribute(attributeIndex).numValues();
		// split the given data according to attributeIndex values
		Instances splittedData[] = splitData(data, attributeIndex);
		for (int i = 0; i < numValues; i++) {
			if (splittedData[i] != null) {
				double probabilty = (double) splittedData[i].size() / (double) data.size();
				if (probabilty != 0) {
					splitInformation += probabilty * Math.log10(probabilty);
				}
			}
		}
		splitInformation *= -1;
		return splitInformation;
	}

	/**
	 * Calculates the entropy of a random variable where all the probabilities
	 * of all of the possible values it can take are given as input.
	 * 
	 * @param probabilities
	 *            - probabilities of all of the possible values of the variable
	 * @return - the entropy
	 */
	private double calcEntropy(double[] probabilities) {
		// calculate entropy according to the formula
		double entropy = 0;
		for (int i = 0; i < probabilities.length; i++) {
			if (probabilities[i] != 0) {
				entropy += probabilities[i] * (Math.log10(probabilities[i]));
			}
		}
		entropy *= -1;
		return entropy;
	}

	/**
	 * Returns an array of probabilities for each class on a given data
	 * 
	 * @param data
	 * @return - probabilities
	 */
	private double[] getProbabilities(Instances data) {
		// an array of size - number of classes
		double[] probabilities = new double[data.numClasses()];
		// counts each appearance of each instance according to it's class value
		for (Instance instance : data) {
			probabilities[(int) instance.classValue()] += 1;
		}

		// set the probability by dividing by the size of all instances in the
		// data
		for (int i = 0; i < probabilities.length; i++) {
			probabilities[i] = probabilities[i] / data.size();
		}

		return probabilities;
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}

	public void setValidation(Instances validation) {
		validationSet = validation;
	}

	@Override
	public double classifyInstance(Instance instance) {
		List<Rule> suitableRules = findMostSuitableRules(instance);
		return findMajorityByRules(suitableRules);
	}

	/**
	 * finds the rules that meet the largest number of consecutive conditions
	 * for instance
	 * 
	 * @param instance
	 *            - instance for which we search most suitable rules
	 * @return - a list of most suitable rules
	 */
	private List<Rule> findMostSuitableRules(Instance instance) {
		List<Rule> suitableRules = new ArrayList<Rule>(rules);

		suitableRules = new ArrayList<Rule>();
		Rule closest = null;
		int counterClosest = 0;
		int counterCurrent = 0;
		for (Rule rule : rules) {
			for (BasicRule basicRule : rule.basicRules) {
				if (basicRule.attributeValue == instance.value(basicRule.attributeIndex)) {
					counterCurrent++;
				} else {
					if (counterCurrent > counterClosest) {
						closest = rule;
						counterClosest = counterCurrent;
						suitableRules = new ArrayList<Rule>();
						suitableRules.add(rule);
					} else if (counterCurrent == counterClosest) {
						suitableRules.add(rule);
					}
					break;
				}
			}
			counterCurrent = 0;
		}

		return suitableRules;

	}

	/**
	 * finds the majority of class values for given rules
	 * 
	 * @param rules
	 *            - the rules for which to find majority
	 * @return - the majority class value
	 */
	private double findMajorityByRules(List<Rule> rules) {
		// counters for the 2 possible class values
		int positives = 0, negative = 0;
		for (Rule rule : rules) {
			if (rule.returnValue == 0) {
				positives++;
			} else {
				negative++;
			}
		}
		if (positives > negative) {
			return 0;
		} else {
			return 1;
		}
	}

	/**
	 * finds the majority of class values for given rules
	 * 
	 * @param rules
	 *            - the rules for which to find majority
	 * @return - the majority class value
	 */
	private double findMajorityByInstances(Instances data) {
		// counters for the 2 possible class values
		int positives = 0, negative = 0;
		for (Instance instance : data) {
			if (instance.classValue() == 0) {
				positives++;
			} else {
				negative++;
			}
		}
		if (positives > negative) {
			return 0;
		} else {
			return 1;
		}
	}

	private Instances[] splitData(Instances data, int attributeIndex) {
		data.sort(attributeIndex);
		Instances splittedData[] = new Instances[data.attribute(attributeIndex).numValues()];
		int firstIndex = 0;
		int lastIndex = 0;
		for (int i = 0; i < data.size(); i++) {
			if (i == data.size() - 1) {
				lastIndex = i;
				splittedData[(int) data.instance(i).value(attributeIndex)] = new Instances(data, firstIndex,
						lastIndex - firstIndex + 1);
				splittedData[(int) data.instance(i).value(attributeIndex)].setClassIndex(data.numAttributes() - 1);
			} else if (data.instance(i).value(attributeIndex) != data.instance(i + 1).value(attributeIndex)) {
				lastIndex = i;
				splittedData[(int) data.instance(i).value(attributeIndex)] = new Instances(data, firstIndex,
						lastIndex - firstIndex + 1);
				splittedData[(int) data.instance(i).value(attributeIndex)].setClassIndex(data.numAttributes() - 1);
				firstIndex = i + 1;
				lastIndex = i + 1;
			}
		}

		return splittedData;
	}

	public int getNumOfRules() {
		return rules.size();
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