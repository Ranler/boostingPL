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

package boostingPL.MR;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import weka.classifiers.Classifier;
import weka.core.Instances;

import boostingPL.MR.io.ClassifierWritable;
import boostingPL.boosting.AdaBoost;
import boostingPL.boosting.InstancesHelper;
import boostingPL.utils.Sort;


public class AdaBoostPLMapper 
	extends Mapper<LongWritable, Text, IntWritable, ClassifierWritable>{
	
	private Instances insts = null;
	
	protected void map(LongWritable key, Text value, Context context) {
		if (insts == null) {
			insts = InstancesHelper.createInstances(value.toString());
		}		
		insts.add(InstancesHelper.createInstance(value.toString(), insts));
	}
	
	protected void cleanup(Context context) throws IOException ,InterruptedException {
		
		int T = Integer.parseInt(context.getConfiguration().get("AdaBoost.numInterations"));
	
		AdaBoost adaBoost = new AdaBoost(insts, T);
		try {
			adaBoost.run();
		} catch (Exception e2) {
			System.out.println("Training Error");
			e2.printStackTrace();
			return;
		}

		double[] corWeights = adaBoost.getClasifiersWeights();
		Classifier[] classifiers = adaBoost.getClassifiers();
		int taskid = context.getTaskAttemptID().getTaskID().getId();

		Sort.sort(classifiers, corWeights);		
		for (int i = 0; i < classifiers.length; i++) {
			context.write(new IntWritable(taskid),
					new ClassifierWritable(classifiers[i], corWeights[i]));
		}
	}

}