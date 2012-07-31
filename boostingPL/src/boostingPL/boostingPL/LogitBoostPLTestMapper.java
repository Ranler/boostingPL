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
import java.util.ArrayList;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.LineReader;

import boostingPL.core.Instance;
import boostingPL.weakclassifier.WeakClassifier;
import boostingPL.weakclassifier.WeakClassifierHelper;

public class LogitBoostPLTestMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable> {
	private ArrayList<WeakClassifier> classifiers;

	protected void setup(Context context) throws java.io.IOException ,InterruptedException {
		String src = context.getConfiguration().get("AdaBoost.ClassifiersFile"); //TODO
	
		classifiers = new ArrayList<WeakClassifier>();
		
		FileSystem hdfs = FileSystem.get(context.getConfiguration());
		FSDataInputStream dis = hdfs.open(new Path(src));
		LineReader in = new LineReader(dis);
		Text line = new Text();
		while(in.readLine(line) > 0){
			String[] items = line.toString().split("|");
			for (int i = 0; i < items.length;) {
				WeakClassifier w = WeakClassifierHelper.newInstance();		
				w.fromString(items[i]);
				classifiers.add(w);
			}
		}
		dis.close();
		in.close();		
	}

	protected void map(LongWritable key, Text value, Context context) throws IOException ,InterruptedException {
		Instance inst = new Instance(value.toString());
		double H = 0;
		for (int i = 0; i < classifiers.size(); i++) {
			H += classifiers.get(i).classifyInstance(inst);
		}
		H = H / classifiers.size();
		
		int hypoth = H >= 0.0 ? +1 : -1;
		if(hypoth == inst.getClassAttr()){
			context.getCounter("Error Rate", "Right number").increment(1);
		}
		else{
			context.getCounter("Error Rate", "Error number").increment(1);
		}
	}
}

