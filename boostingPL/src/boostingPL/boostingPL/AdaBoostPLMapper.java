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

package boostingPL.boostingPL;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import boostingPL.boosting.AdaBoost;
import boostingPL.core.Instance;
import boostingPL.core.Instances;
import boostingPL.mr.io.WeakClassifierArrayWritable;
import boostingPL.weakclassifier.WeakClassifier;


public class AdaBoostPLMapper extends Mapper<LongWritable, Text, Text, WeakClassifierArrayWritable>{
	private Instances insts;
	
	protected void setup(Context context) throws IOException ,InterruptedException {
		insts = new Instances();
	}
	
	protected void map(LongWritable key, Text value, Context context) throws IOException ,InterruptedException {
		insts.addInstance(new Instance(value.toString()));
	}
	
	protected void cleanup(Context context) throws IOException ,InterruptedException {
		int T = Integer.parseInt(context.getConfiguration().get("AdaBoost.numInterations"));
		AdaBoost adaBoost = new AdaBoost(insts, T);
		adaBoost.run();
		
		sort(adaBoost.getWeakClassifiers());
		
		context.write(new Text("H"), new WeakClassifierArrayWritable(adaBoost.getWeakClassifiers()));
	}
	
	private void sort(WeakClassifier[] classifiers) {
		quickSort(classifiers, 0, classifiers.length-1);
	}
	
	private void quickSort(WeakClassifier[] classifiers, int left, int right) {
		WeakClassifier temp;
		if(left < right) {
			int i = left;
			int j = right + 1;
			while(true) {
				while(i+1 < classifiers.length 
						&& classifiers[++i].getCorWeight() 
						< classifiers[left].getCorWeight());
				while(j-1 > -1
						&& classifiers[--j].getCorWeight() 
						> classifiers[left].getCorWeight());
				
				if (i >= j) {
					break;
				}
				
				temp = classifiers[i];
				classifiers[i] = classifiers[j];
				classifiers[j] = temp;
			}
	
			temp = classifiers[left];
			classifiers[left] = classifiers[j];
			classifiers[j] = temp;			
			
			quickSort(classifiers, left, j-1);
			quickSort(classifiers, j+1, right);
		}
	}
}
