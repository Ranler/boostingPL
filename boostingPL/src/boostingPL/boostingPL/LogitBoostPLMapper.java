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

import boostingPL.boosting.LogitBoost;
import boostingPL.core.Instance;
import boostingPL.core.Instances;
import boostingPL.mr.io.WeakClassifierArrayWritable;
import boostingPL.utils.Sort;

public class LogitBoostPLMapper extends Mapper<LongWritable, Text, Text, WeakClassifierArrayWritable>{
	
	private Instances insts;
	
	protected void setup(Context context) throws IOException ,InterruptedException {
		insts = new Instances();
	}
	
	protected void map(LongWritable key, Text value, Context context) throws IOException ,InterruptedException {
		insts.addInstance(new Instance(value.toString()));
	}
	
	protected void cleanup(Context context) throws IOException ,InterruptedException {
		int T = Integer.parseInt(context.getConfiguration().get("AdaBoost.numInterations"));
		LogitBoost logitboost = new LogitBoost(insts, T);
		logitboost.run();
		
		Sort.sort(logitboost.getWeakClassifiers(), logitboost.getErrorRates());
		
		context.write(null, new WeakClassifierArrayWritable(logitboost.getWeakClassifiers()));
		
	}

}
