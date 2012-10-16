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
	
	public AdaBoostPL(Classifier[][] classifiers, double[][] corWeights) {
		this.classifiers = classifiers;		
		
		this.corWeights = new double[corWeights[0].length];
		for (int i = 0; i < corWeights.length; i++) {
			for (int j = 0; j < corWeights[i].length; j++) {
				this.corWeights[j] += corWeights[i][j];
			}
			System.out.println();
		}
	}
	
	@Override
	public void buildClassifier(Instances insts) throws Exception {}

	@Override
	public double classifyInstance(Instance inst) throws Exception {
		int classNum = inst.dataset().classAttribute().numValues();
		double[] H = new double[classNum];
		for (int j = 0; j < corWeights.length; j++) {
			int classValue = merge(inst, j, classNum);
			if (classValue >= 0) {
				H[classValue] += corWeights[j];
			}

		}
		return (double)maxIdx(H);
	}

	@Override
	public double[] distributionForInstance(Instance inst) throws Exception {
		int classNum = inst.dataset().classAttribute().numValues();
		double[] H = new double[classNum];
		double sum = 0;
		for (int j = 0; j < corWeights.length; j++) {
			int classValue = merge(inst, j, classNum);
			if (classValue >= 0) {
				H[classValue] += corWeights[j];
				sum += corWeights[j];
			}	
		}

		// normalize
		for (int i = 0; i < H.length; i++) {
			H[i] /= sum;
		}
		return H;
	}
	
	private int merge(Instance inst, int round, int classNum) throws Exception {
		int[] sum = new int[classNum];
		for (int i = 0; i < classifiers.length; i++) {
			int classIdx = (int)classifiers[i][round].classifyInstance(inst);
			sum[classIdx] += 1;
		}
		
		return maxIdx(sum);
	}
	
	private int maxIdx(int[] a) {
		int max = -1;
		int maxIdx = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > max) {
				maxIdx = i;
				max = a[i];
			}
			else if (a[i] > 0 && a[i] == max) {
				// more than two classes have same vote  
				return -1;
			}
		}
		return maxIdx;
	}
	
	private int maxIdx(double[] a) {
		double max = -1;
		int maxIdx = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > max) {
				maxIdx = i;
				max = a[i];
			}
			else if (a[i] > 0 && a[i] == max) {
				// more than two classes have same vote  
				return -1;
			}
		}
		return maxIdx;
	}	
	
	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
}