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

import boostingPL.core.Instance;
import boostingPL.core.Instances;
import boostingPL.weakclassifier.WeakClassifier;
import boostingPL.weakclassifier.WeakClassifierHelper;

/**
 * Logit Adaptive Boosting
 * 
 * @author Ranler Cao  findfunaax@gmail.com
 *
 */
public class LogitBoost {
	
	private Instances insts;
	
	private int numInterations;
	
	private WeakClassifier[] classifiers;
	private double[] errorRates;
	

	public LogitBoost(Instances insts, int numInterations) {
		this.insts = insts;
		this.numInterations = numInterations;
		this.classifiers = new WeakClassifier[numInterations];
		this.errorRates = new double[numInterations];
	}
	
	public boolean run() {
		double[] F = new double[insts.numInstances()];
		double[] p = new double[insts.numInstances()];
		for (int i = 0; i < p.length; i++) {
			F[i] = 0.0;
			p[i] = 0.5;
		}
		
		double[] z = new double[insts.numInstances()];
		double[] w = new double[insts.numInstances()];
		for (int t = 0; t < numInterations; t++) {
			for (int i = 0; i < w.length; i++) {
				// {-1, 1} => {0, 1}
				int y_star = insts.getDataSets().get(i).getClassAttr() == -1 ? 0 : 1;
				
				z[i] = (y_star - p[i]) / (p[i] * (1 - p[i]));
				w[i] = p[i] * (1 - p[i]);
			}
			
			classifiers[t] = WeakClassifierHelper.newInstance();
			classifiers[t].learnWeakClassifier(insts, w, z);
			
			for (int i = 0; i < w.length; i++) {
				Instance inst = insts.getInstance(i);
				int hyp = classifiers[t].classifyInstance(inst);
				
				F[i] = F[i] + 0.5 * hyp;
				p[i] = Math.pow(Math.E, F[i]) / 
						(Math.pow(Math.E, F[i]) + Math.pow(Math.E, -F[i]));
				
				// unweighted error rate
				if(hyp != inst.getClassAttr()){
					errorRates[i] += inst.getClassAttr();
				}
			}
		}
		return true;
	}
	
	public int classifyInstance(Instance inst) {
		double H = 0.0;
		for (int t = 0; t < numInterations; ++t){
				H += classifiers[t].classifyInstance(inst);
		}
		return H >= 0.0 ? +1 : -1;
	}

	public WeakClassifier[] getWeakClassifiers() {
		return classifiers;
	}
	
	public double[] getErrorRates() {
		return errorRates;
	}
	
}
