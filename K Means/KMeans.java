/**
 * @author Tomer Korbinsky & Arad Zaltsberger
 * ID's: 203021720, 312533342
 */
package HomeWork7;

import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public class KMeans {

	private int m_KNumberOfClusters;
	private Cluster[] m_Clusters;
	private static final int presetNumberOfIterations = 40;

	/**
	 * Set number of clusters.
	 * 
	 * @param m_KNumberOfClusters
	 */
	public void setKNumberOfClusters(int numberOfClusters) {
		this.m_KNumberOfClusters = numberOfClusters;
	}

	/**
	 * This method is building the KMeans object. It should initialize centroids
	 * (by calling initializeCentroids) and run the K-Means algorithm (which
	 * means to call findKMeansCentroids methods).
	 * 
	 * @param data
	 */
	public void buildClusterModel(Instances data) {
		initializeCentroids(data);
		findKMeansCentroids(data);
	}

	/**
	 * Initialize the centroids by selecting k random instances from the
	 * training set and setting the centroids to be those instances.
	 * 
	 * @param data
	 */
	public void initializeCentroids(Instances data) {
		Random random = new Random();
		Instances shuffledData = new Instances(data, 0, data.size());
		shuffledData.randomize(random);
		m_Clusters = new Cluster[m_KNumberOfClusters];
		for (int i = 0; i < m_Clusters.length; i++) {
			// in the beginning, instance i is the centroid of the cluster i
			m_Clusters[i] = new Cluster(data, shuffledData.instance(i));
		}
	}

	/**
	 * Should find and store the centroids according to the KMeans algorithm.
	 * Your stopping condition will be a preset number of iterations - 40. Use
	 * one or any combination of these methods to determine when to stop
	 * iterating.
	 * 
	 * @param data
	 */
	public void findKMeansCentroids(Instances data) {
		for (int i = 0; i < presetNumberOfIterations; i++) {
			for (Instance instance : data) {
				int closestCentroidClusterIndex = findClosestCentroid(instance);
				m_Clusters[closestCentroidClusterIndex].addInstance(instance);
			}
			/*
			 * Print the Avg WSSSE only for provide a graphical representation
			 * of the total error as a function of the iteration number, in K =
			 * 5
			 */
			if (m_KNumberOfClusters == 5) {
				double avgWSSSE = calcAvgWSSSE(data);
				System.out.println("Avg WSSSE in iteration number	" + (i + 1) + " =	" + avgWSSSE);
			}
			updateClustersCentroids();
			clearClusters();
		}
	}

	/**
	 * Update each cluster's centroid according to the new cluster instances.
	 */
	private void updateClustersCentroids() {
		for (Cluster cluster : m_Clusters) {
			cluster.setCentroid(findClusterMean(cluster));
		}
	}

	/**
	 * Find the mean of the given cluster.
	 * 
	 * @param cluster
	 * @return
	 */
	private Instance findClusterMean(Cluster cluster) {
		double[] meanCentroidAttributes = new double[cluster.getCentroid().numAttributes()];
		for (int i = 0; i < meanCentroidAttributes.length; i++) {
			double meanAttributeValue = 0;
			for (Instance instance : cluster.getInstances()) {
				meanAttributeValue += instance.value(i);
			}
			meanAttributeValue /= cluster.getInstances().size();
			meanCentroidAttributes[i] = meanAttributeValue;
		}
		Instance updatedCentroid = cluster.getCentroid();
		for (int i = 0; i < meanCentroidAttributes.length; i++) {
			updatedCentroid.setValue(i, meanCentroidAttributes[i]);
		}
		return updatedCentroid;
	}

	/**
	 * Clear all instances of each cluster.
	 */
	private void clearClusters() {
		for (Cluster cluster : m_Clusters) {
			cluster.clearInstances();
		}
	}

	/**
	 * Should calculate the squared distance between the input instance and the
	 * input centroid.
	 * 
	 * @param instanceFromdataSet
	 * @param centroid
	 * @return
	 */
	public double calcSquaredDistanceFromCentroid(Instance instanceFromdataSet, Instance centroid) {
		double squaredDistance = 0;
		for (int i = 0; i < instanceFromdataSet.numAttributes(); i++) {
			squaredDistance += Math.pow(instanceFromdataSet.value(i) - centroid.value(i), 2);
		}
		return squaredDistance;
	}

	/**
	 * finds the index of the closest centroid to the input instance
	 * 
	 * @param instance
	 * @return
	 */
	public int findClosestCentroid(Instance instance) {
		int closestCentroidClusterIndex = -1;
		double closestCentroidDistance = Double.MAX_VALUE;
		for (int j = 0; j < m_Clusters.length; j++) {
			int currentCentroidClusterIndex = j;
			double currentDistance = calcSquaredDistanceFromCentroid(instance, m_Clusters[j].getCentroid());
			if (currentDistance < closestCentroidDistance) {
				closestCentroidDistance = currentDistance;
				closestCentroidClusterIndex = currentCentroidClusterIndex;
			}
		}
		return closestCentroidClusterIndex;
	}

	/**
	 * should replace every instance in Instances by the centroid to which it is
	 * assigned (closest centroid) and return the new Instances object.
	 * 
	 * @param data
	 * @return
	 */
	public Instances quantize(Instances data) {
		Instances quantizedData = new Instances(data, 0, 0);
		for (Instance instance : data) {
			int clusterIndex = findClosestCentroid(instance);
			quantizedData.add(m_Clusters[clusterIndex].getCentroid());
		}
		return quantizedData;
	}

	/**
	 * should be the average within set sum of squared errors. That is it should
	 * calculate the average squared distance of every instance from the
	 * centroid to which it is assigned. This is Tr(Sc) from class, divided by
	 * the number of instances. Return the double value of the WSSSE.
	 * 
	 * @param data
	 * @return
	 */
	public double calcAvgWSSSE(Instances data) {
		double avgWSSSE = 0;
		for (Cluster cluster : m_Clusters) {
			for (Instance instance : cluster.getInstances()) {
				avgWSSSE += calcSquaredDistanceFromCentroid(instance, cluster.getCentroid());
			}

		}
		avgWSSSE /= data.numInstances();
		return avgWSSSE;
	}
}

/**
 * An object that represents a cluster.
 */
class Cluster {

	private Instances instances;
	private Instance centroid;

	public Cluster(Instances data, Instance centroid) {
		this.instances = new Instances(data, 0, 0);
		this.centroid = centroid;
	}

	public void clearInstances() {
		instances = new Instances(instances, 0, 0);
	}

	public void addInstance(Instance instance) {
		instances.add(instance);
	}

	public void setCentroid(Instance centroid) {
		this.centroid = centroid;
	}

	public Instance getCentroid() {
		return centroid;
	}

	public Instances getInstances() {
		return instances;
	}
}