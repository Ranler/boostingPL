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
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;


/**
 * Adaptive Boosting
 * 
 * @author Ranler Cao  findfunaax@gmail.com
 *
 */
public class AdaBoost {

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
	}
	
	public boolean run() throws Exception {
		// initialize instance's weight
		int numInstances = insts.numInstances();
		for (int i = 0; i < numInstances; i++) {
			double tweight = 1.0/numInstances;
			insts.instance(i).setWeight(tweight);
		}
		//System.out.println("instances weights total: " + insts.sumOfWeights());
		
		for (int t = 0; t < numIterations; ++t){
			classifiers[t] = ClassifiersHelper.newInstance("DecisionStump");
			classifiers[t].buildClassifier(insts);
			
			double e = weightError(t);
			if(e >= 0.5) {
				System.out.println("Error: epsilon > 0.5");
				return false;
			}
			cweights[t] = 0.5 * Math.log((1-e)/e) / Math.log(Math.E);
			System.out.println("Round = " + t
					+ "\t ErrorRate = " + e 
					+ "\t CorWeights = " + cweights[t]);
			
			for (int i = 0; i < insts.numInstances(); i++) {
				Instance inst = insts.instance(i);
				if (classifiers[t].classifyInstance(inst) != inst.classValue()) {
					inst.setWeight(inst.weight() / (2 * e));
				} else {
					inst.setWeight(inst.weight() / (2 * (1-e)));
				}
			}
		}
		return true;
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
}
