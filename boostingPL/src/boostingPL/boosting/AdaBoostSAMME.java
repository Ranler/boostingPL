package boostingPL.boosting;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class AdaBoostSAMME {

	/** training instances */
	private Instances insts;
	
	/** the number of iteration */
	private int numIterations;
	
	/** weak classifiers */
	private Classifier[] classifiers;
	
	/** weights for all weak classifiers */
	private double[] cweights;
	
	
	public AdaBoostSAMME(Instances insts, int numInterations) {
		this.insts = insts;
		this.numIterations = numInterations;
		this.classifiers = new Classifier[numInterations];
		this.cweights = new double[numInterations];
		
		// initialize instance's weight
		final int numInstances = insts.numInstances();		
		for (int i = 0; i < numInstances; i++) {
			double tweight = 1.0/numInstances;
			insts.instance(i).setWeight(tweight);
		}
		//System.out.println("instances weights total: " + insts.sumOfWeights());

	}
	
	public void run(int t) throws Exception {
		if (t >= numIterations) {
			return;
		}
		
		classifiers[t] = ClassifiersHelper.newInstance("DecisionStump");
		//classifiers[t] = ClassifiersHelper.newInstance("C4.5");
		classifiers[t].buildClassifier(insts);

		double e = weightError(t);
		final int numClasses = insts.classAttribute().numValues();
		double maxe = 1 - 1.0 / numClasses;
		if (e >= maxe) {
			System.out.println("Error: error rate = " + e + ", >= " + maxe);
			// saveInstance(insts);
			throw new Exception("error rate > " + maxe);
		}

		cweights[t] = Math.log((1 - e) / e) + Math.log(numClasses - 1);
		System.out.println("Round = " + t + "\tErrorRate = " + e
				+ "\tCWeight = " + cweights[t] + "\texpCWeight = "
				+ Math.exp(cweights[t]));

		double expCWeight = Math.exp(cweights[t]);
		for (int i = 0; i < insts.numInstances(); i++) {
			Instance inst = insts.instance(i);
			if (classifiers[t].classifyInstance(inst) != inst.classValue()) {
				inst.setWeight(inst.weight() * expCWeight);
			}
		}

		double weightSum = insts.sumOfWeights();
		for (int i = 0; i < insts.numInstances(); i++) {
			Instance inst = insts.instance(i);
			inst.setWeight(inst.weight() / weightSum);
		}

	}

	public Classifier[] getClassifiers() {
		return classifiers;
	}

	public double[] getClasifiersWeights() {
		return cweights;
	}
	
	private double weightError(int t) throws Exception{
		// evaluate all instances
		Evaluation eval = new Evaluation(insts);
		eval.evaluateModel(classifiers[t], insts);
		return eval.errorRate();
	}
	
	public double classifyInstance(Instance inst) throws Exception {
		int classNum = inst.dataset().classAttribute().numValues();
		double[] H = new double[classNum];
		for (int j = 0; j < cweights.length; j++) {
			int classValue = (int)classifiers[j].classifyInstance(inst);
			if (classValue >= 0) {
				H[classValue] += cweights[j];
			}
		}
		return (double)maxIdx(H);
	}
	
	private int maxIdx(double[] a) {
		double max = -1;
		int maxIdx = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > max) {
				maxIdx = i;
				max = a[i];
			}
			else if (a[i] == max) {
				// at least two classes have same vote  
				return -1;
			}
		}
		return maxIdx;
	}	
	
	public static void main(String[] args) throws Exception {
		java.io.File inputFile = new java.io.File("/home/aax/xpShareSpace/boostingPL/RDG1-100-5.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances training = atf.getDataSet();
		training.setClassIndex(training.numAttributes()-1);
		
		AdaBoostSAMME samme = new AdaBoostSAMME(training, 100);
		for (int t = 0; t < 100; t++) {
			samme.run(t);			
		}

		
		int right = 0;
		for (int i = 0; i < training.numInstances(); i++) {
			Instance inst = training.instance(i);
			if (samme.classifyInstance(inst) == inst.classValue()) {
				right++;
			}
		}
		System.out.println(right);
		System.out.println((double)right/training.numInstances());
	}
}
