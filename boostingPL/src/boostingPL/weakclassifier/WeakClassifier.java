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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import boostingPL.core.Instance;
import boostingPL.core.Instances;

public abstract class WeakClassifier implements Writable {
	
	/** corresponding weight of weak classifier, needed for AdaBoost */
	private double corWeight;
	
	public double getCorWeight() {
		return corWeight;
	}
	
	public void setCorWeight(double corWeight) {
		this.corWeight = corWeight;
	}	
	
	@Override
	public void readFields(DataInput in) throws IOException {
		corWeight = in.readDouble();
	} 
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(corWeight);
	}
	
	@Override
	public String toString() {
		return corWeight + "";
	}
	
	public void fromString(String s) {
		corWeight = Double.parseDouble(s);
	}
	
	/**
	 * Train this weak classifier.
	 * 
	 * @param insts training samples
	 * @param weights training samples' weights
	 * @param workResp training samples' working response, needed for LogitBoost
	 */
	public abstract void learnWeakClassifier(Instances insts, double[] weights, double[] workResp);
	
	/**
	 * Classifies the given test instance.
	 * 
	 * @param inst the instance to be classified
	 * @return the predicted most likely class for the instance, 
	 * must be +1 or -1 for AdaBoost
	 */
	public abstract int classifyInstance(Instance inst);
}
