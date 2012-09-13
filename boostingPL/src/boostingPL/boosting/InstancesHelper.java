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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;

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
	
	/**
	 * create instances header from metadata,
	 * the metadata like this:
	 * 
	 *   <br/>
	 *   <p>attributesNum:100</p>
	 *   <p>classes:+1,-1</p>
	 *   <br/>
	 * 
	 * @param in
	 * @return
	 * @throws IOException
	 */
	public static Instances createInstancesFromMetadata(LineReader in) throws IOException {
		int attributesNum = 0;
		ArrayList<Attribute> attInfo = null;
		List<String> classItems = null;
		
		Text line = new Text();
		while(in.readLine(line) > 0){
			String sline = line.toString();
			if (sline.startsWith("attributesNum:")) {
				attributesNum = Integer.parseInt(sline.substring(14));
				attInfo = new ArrayList<Attribute>(attributesNum+1);
				for (int i = 0; i < attributesNum; i++) {
					attInfo.add(new Attribute("attr"+i));
				}
				
				System.out.println("AttributeNum:"+attributesNum);
			}
			else if (sline.startsWith("classes:")) {
				String classes = sline.substring(8);
				String[] citems = classes.split(",");
				classItems = new ArrayList<String>(citems.length);
				for (String s : citems) {
					classItems.add(s);
				}
				
				System.out.println("classes:"+classes);
			}
		}
		
		attInfo.add(new Attribute("class", classItems));
		Instances insts = new Instances("BoostingPL-dataset", attInfo, 0);
		insts.setClassIndex(insts.numAttributes()-1);

		return insts;		
	}
	
	/**
	 * create instances header from a instance
	 * 
	 * @param instance
	 * @return instances
	 */
	public static Instances createInstances(String text) {
		String[] items = text.split(" ");
		
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		for (int i = 0; i < items.length - 1; i++) {
			attInfo.add(new Attribute("attr"+i));
		}
		
		List<String> classItems = new ArrayList<String>(2);
		classItems.add("1");
		classItems.add("-1");		
		attInfo.add(new Attribute("class", classItems));
		Instances insts = new Instances("BoostingPL-dataset", attInfo, 0);
		insts.setClassIndex(insts.numAttributes()-1);

		return insts;
	}		
}