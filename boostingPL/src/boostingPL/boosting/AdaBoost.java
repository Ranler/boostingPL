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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.Math;



import boostingPL.core.Instance;
import boostingPL.core.Instances;
import boostingPL.weakclassifier.WeakClassifier;
import boostingPL.weakclassifier.WeakClassifierHelper;

/**
 * Adaptive Boosting
 * 
 * @author Ranler Cao  findfunaax@gmail.com
 *
 */
public class AdaBoost {

	private Instances insts;
	private int numInterations;
	
	private WeakClassifier[] classifiers;
	
	public AdaBoost(Instances insts, int numInterations){
		this.insts = insts;
		this.numInterations = numInterations;
		this.classifiers = new WeakClassifier[numInterations];
	}
	
	public boolean run() {
		// set instance init weight
		int numInstances = insts.numInstances();
		double[] weights = new double[numInstances];
		for (int i = 0; i < numInstances; i++) {
			weights[i] = 1.0/numInstances;
		}		
		
		for (int t = 0; t < numInterations; ++t){
			classifiers[t] = WeakClassifierHelper.newInstance();
			classifiers[t].learnWeakClassifier(insts, weights);
			
			double e = weightError(weights, t);
			if(e >= 0.5) {
				//System.out.println("Error: epsilon > 0.5");
				return false;
			}
			classifiers[t].setCorWeight(0.5 * Math.log((1-e)/e) / Math.log(Math.E));
			System.out.println("Round = " + t
					+ "\t ErrorRate = " + e 
					+ "\t CorWeights = " + classifiers[t].getCorWeight());
			
			for (int i = 0; i < insts.numInstances(); i++) {
				Instance inst = insts.getInstance(i);
				if (classifiers[t].classifyInstance(inst) != inst.getClassAttr()) {
					weights[i] = weights[i] / (2 * e);
				} else {
					weights[i] = weights[i] / (2 * (1-e));
				}
			}
		}
		return true;
	}
	
	public int classifyInstance(Instance inst){
		double H = 0.0;
		for (int t = 0; t < numInterations; ++t){
				H += classifiers[t].getCorWeight() * classifiers[t].classifyInstance(inst);
		}
		
		if (H >= 0.0) {
			return +1;
		}
		else
			return -1;
	}

	public WeakClassifier[] getWeakClassifiers(){
		return classifiers;
	}

	private double weightError(double[] weights, int t){
		double e = 0.0;
		for (int i = 0; i < insts.numInstances(); i++) {
			Instance inst = insts.getInstance(i);
			if(inst.getClassAttr() != classifiers[t].classifyInstance(inst)){
				e += weights[i];
			} 
		}
		return e;
	}
	

	public static void main(String[] args) throws IOException{
		Instances insts = new Instances();
		
		FileReader reader = new FileReader(args[0]);
		BufferedReader br = new BufferedReader(reader);
		String line;
		while((line = br.readLine()) != null){
			insts.addInstance(new Instance(line));
		}
		br.close();
		reader.close();
		
		System.out.println("Instances Number = " + insts.numInstances());
		System.out.println("Attributes Number = " + insts.numAttributes());
		System.out.println("Start AdaBoost...");
		WeakClassifierHelper.setClassifierClass("DecisionStump");
		AdaBoost adaBoosting = new AdaBoost(insts, 100);  // TODO
		adaBoosting.run();
		System.out.println("AdaBoost Train Over");

		int rightCount = 0;
		for (Instance inst : insts.getDataSets()) {
			if (adaBoosting.classifyInstance(inst) == inst.getClassAttr()){
				rightCount++;
			}
		}
		System.out.println(rightCount + "/" + insts.numInstances());
		System.out.println(rightCount * 1.0 / insts.numInstances());
	}

}
