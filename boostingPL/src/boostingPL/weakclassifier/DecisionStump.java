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

package boostingPL.weakclassifier;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.FileReader;
import java.io.IOException;

import boostingPL.core.Instance;
import boostingPL.core.Instances;

public class DecisionStump extends WeakClassifier {
	
	/** parameters */
	private int attIndex;
	private double splitPoint;
	/** class = 1*classFix if attribute[attIndex] < splitPoint */
	private int classFix;
	
	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		attIndex = in.readInt();
		splitPoint = in.readDouble();
		classFix = in.readInt();
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		out.writeInt(attIndex);
		out.writeDouble(splitPoint);
		out.writeInt(classFix);
	}
	
	@Override
	public String toString() {
		return super.toString() + "," + attIndex + "," + splitPoint + "," + classFix;
	}
	
	public void fromString(String s) {
		String[] paras = s.split(",");
		super.fromString(paras[0]);
		attIndex = Integer.parseInt(paras[1]);
		splitPoint = Double.parseDouble(paras[2]);
		classFix = Integer.parseInt(paras[3]);
	}
	
	@Override
	public void learnWeakClassifier(Instances insts, double[] weights, double[] workingResp) {
		// TODO 重写
		double[][] samples = new double[insts.numInstances()][insts.numAttributes()];
		for (int i = 0; i < samples.length; i++) {
			samples[i] = insts.getInstance(i).getAttr();
		}
		int[] flags = new int[insts.numInstances()];
		for (int i = 0; i < flags.length; i++) {
			flags[i] = insts.getInstance(i).getClassAttr();
		}
		
		double minAllErrorRate = Double.MAX_VALUE;
		for (int j = 0; j < samples[0].length; j++) {
			double[] sortedVers = new double[samples.length];
			double[] sortedWeights = weights.clone();
			int[] sortedFlags = flags.clone();
			
			double errorRate = 0; // 1 if attribute[attIndex] < splitPoint
			double errorRateR = 0; // -1 if attribute[attIndex] < splitPoint
			for (int i = 0; i < samples.length; i++) {
				sortedVers[i] = samples[i][j];
				if (flags[i] == 1) errorRate += weights[i];
				else errorRateR += weights[i];
			}
			shellSort(sortedVers, sortedWeights, sortedFlags);
			
			double minErrorRate = Double.MAX_VALUE;
			int minFlagFix = 1;
			double minFlagFalseTheta = 0;
			// <j属于左边1类， >= j 属于右边-1类
			for (int k = 1; k < sortedVers.length; k++) {
				if (sortedFlags[k] == 1) {
					errorRate -= sortedWeights[k];
					errorRateR += sortedWeights[k];
				}else{
					errorRate += sortedWeights[k];
					errorRateR -= sortedWeights[k];
				}
				
				if (errorRate < minErrorRate){
					minErrorRate = errorRate;
					minFlagFalseTheta = sortedVers[k];
					minFlagFix = 1;
				}
				if (errorRateR < minErrorRate){
					minErrorRate = errorRateR;
					minFlagFalseTheta = sortedVers[k];
					minFlagFix = -1;
				}				
			}
			
			if (minErrorRate < minAllErrorRate) {
				minAllErrorRate = minErrorRate;
				this.attIndex = j;
				this.splitPoint = minFlagFalseTheta;
				this.classFix = minFlagFix;
			}
		}		
	}
	
	@Override
	public int classifyInstance(Instance inst) {
		int h;
		if(inst.getAttr()[attIndex] < splitPoint){
			h = 1 * classFix;
		}else{ 
			h = -1 * classFix;
		}
		return h;
	}
	
	private void shellSort(double[] x, double[] y, int[] z){
		// sort by x
		for (int inc = x.length/2; inc > 0; inc/=2) {
			for (int i = 0; i < x.length; i++) {
				double tempX = x[i];
				double tempY = y[i];
				int tempZ = z[i];
				int j = 0;
				for (j = i; j >= inc; j-=inc) {
					if (tempX < x[j-inc]){
						x[j] = x[j-inc];
						y[j] = y[j-inc];
						z[j] = z[j-inc];
					}else{
						break;
					}
				}
				x[j] = tempX;
				y[j] = tempY;
				z[j] = tempZ;
			}
		}
	}
	
	public static void main(String[] args) throws IOException {
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
		System.out.println("Start DecisionStump...");
		
		double[] weights = new double[insts.numInstances()];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = 1.0;
		}
		DecisionStump ds = new DecisionStump();
		ds.learnWeakClassifier(insts, weights, null);
		System.out.println("DecisionStump Train Over");

		int rightCount = 0;
		int class1 = 0;
		for (Instance inst : insts.getDataSets()) {
			if (ds.classifyInstance(inst) == inst.getClassAttr()){
				rightCount++;
				if (inst.getClassAttr() == 1) {
					class1 ++;
				}
			}
		}
		System.out.println(rightCount + "/" + insts.numInstances());
		System.out.println(rightCount * 1.0 / insts.numInstances());
		System.out.println(class1);
	}		
}
