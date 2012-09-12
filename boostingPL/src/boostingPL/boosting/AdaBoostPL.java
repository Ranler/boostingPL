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

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class AdaBoostPL implements Classifier {

	private Classifier[][] classifiers;
	private double[] corWeights;
	
	public AdaBoostPL(Classifier[][] classifiers, double[] corWeights) {
		this.classifiers = classifiers;
		this.corWeights = corWeights;
	}
	
	@Override
	public void buildClassifier(Instances insts) throws Exception {}

	@Override
	public double classifyInstance(Instance inst) throws Exception {
		double[] H = new double[2];
		for (int j = 0; j < corWeights.length; j++) {
			int cv = merge(inst, j);
			if (cv >= 0) {
				H[merge(inst, j)] += corWeights[j];
			}
		}
		int pred = H[0] > H[1] ? 0 : 1;
		return (double)pred;
	}

	private int merge(Instance inst, int round) throws Exception{
		int[] sum = new int[2];
		for (int i = 0; i < classifiers.length; i++) {
			sum[(int)classifiers[i][round].classifyInstance(inst)] += 1;
		}
		
		if(sum[0] == sum[1]) return -1;
		return sum[0] > sum[1] ? 0 : 1;
	}
	
	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] distributionForInstance(Instance inst) throws Exception {
		double[] H = new double[2];
		for (int j = 0; j < corWeights.length; j++) {
			int cv = merge(inst, j);
			if (cv >= 0) {
				H[merge(inst, j)] += corWeights[j];
			}
		}
		
		double sum = H[0] + H[1];
		H[0] = H[0] / sum;
		H[1] = H[1] / sum;
		//int pred = H[0] > H[1] ? 0 : 1;
		return H;
	}

}