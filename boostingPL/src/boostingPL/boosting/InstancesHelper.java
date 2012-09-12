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

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class InstancesHelper {
	
	public static Instance createInstance(String text, Instances insts) {
		// numeric attributes
		String[] items = text.split(" ");		
		double[] ds = new double[items.length];
		for (int i = 0; i < ds.length-1; i++) {
			ds[i] = Double.parseDouble(items[i]);
		}
		
		// nominal class attribute
		ds[items.length-1] = insts.classAttribute().indexOfValue(items[items.length-1]);
		
		Instance inst = new DenseInstance(1, ds);
		inst.setDataset(insts);
		return inst;
	}
	
	public static Instances createInstances(String text) {
		String[] items = text.split(" ");
		
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		for (int i = 0; i < items.length - 1; i++) {
			attInfo.add(new Attribute("attr"+i));
		}
		
		// TODO read this from meta data
		List<String> classItems = new ArrayList<String>(2);
		classItems.add("1");
		classItems.add("-1");		
		attInfo.add(new Attribute("class", classItems));
		Instances insts = new Instances("BoostingPL-dataset", attInfo, 0);
		insts.setClassIndex(insts.numAttributes()-1);

		return insts;
	}		
}