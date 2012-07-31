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

package boostingPL.core;

import java.util.ArrayList;

public class Instances {
	private ArrayList<Instance> datasets;
	
	public Instances() {
		this.datasets = new ArrayList<Instance>();
	}

	public void addInstance(Instance inst) {
		datasets.add(inst);
	}
	
	public int numInstances() {
		return datasets.size();
	}
	
	public int numAttributes() {
		if (datasets != null && datasets.size() > 1) {
			return datasets.get(0).attrNum();
		}
		return 0;
	}
	
	public ArrayList<Instance> getDataSets() {
		return datasets;
	}
	
	public Instance getInstance(int i) {
		return datasets.get(i);
	}

}
