/*
 *   BoostingPL - Scalable and Parallel Boosting with MapReduce 
 *   Copyright (C) 2012  Ranler Cao  findfunaax@gmail.com
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   
 */

package boostingPL.boosting;

import java.lang.Math;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

/**
 * Adaptive Boosting
 * 
 * @author Ranler Cao  findfunaax@gmail.com
 *
 */
public class AdaBoost implements Boosting{

	/** training instances */
	private Instances insts;
	
	/** the number of iteration */
	private int numIterations;
	
	/** weak classifiers */
	private Classifier[] classifiers;
	
	/** weights for all weak classifiers */
	private double[] cweights;
	
	
	public AdaBoost(Instances insts, int numInterations) {
		this.insts = insts;
		this.numIterations = numInterations;
		this.classifiers = new Classifier[numInterations];
		this.cweights = new double[numInterations];

		// initialize instance's weight
		int numInstances = insts.numInstances();
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
		
		classifiers[t] = ClassifiersFactory.newInstance("DecisionStump");
		//classifiers[t] = ClassifiersHelper.newInstance("C4.5");
		classifiers[t].buildClassifier(insts);
			
		double e = weightError(t);
		if(e >= 0.5) {
			System.out.println("Error: error rate = " + e + ", >= 0.5");
			throw new Exception("error rate > 0.5");
		}
			
		cweights[t] = 0.5 * Math.log((1-e)/e) / Math.log(Math.E);
		System.out.println("Round = " + t
				+ "\t ErrorRate = " + e 
				+ "\t\t Weights = " + cweights[t]);
			
		for (int i = 0; i < insts.numInstances(); i++) {
			Instance inst = insts.instance(i);
			if (classifiers[t].classifyInstance(inst) != inst.classValue()) {
				inst.setWeight(inst.weight() / (2 * e));
			} else {
				inst.setWeight(inst.weight() / (2 * (1-e)));
			}
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
		java.io.File inputFile = new java.io.File("/home/aax/xpShareSpace/boostingPL/RDG2-100-2.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances training = atf.getDataSet();
		training.setClassIndex(training.numAttributes()-1);
		
		AdaBoost adaBoost = new AdaBoost(training, 100);
		for (int t = 0; t < 100; t++) {
			adaBoost.run(t);			
		}

		
		int right = 0;
		for (int i = 0; i < training.numInstances(); i++) {
			Instance inst = training.instance(i);
			if (adaBoost.classifyInstance(inst) == inst.classValue()) {
				right++;
			}
		}
		System.out.println(right);
		System.out.println((double)right/training.numInstances());		
	}
}
