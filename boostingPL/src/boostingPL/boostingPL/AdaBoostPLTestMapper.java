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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.LineReader;

//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import boostingPL.core.Instance;
import boostingPL.weakclassifier.WeakClassifier;
import boostingPL.weakclassifier.WeakClassifierHelper;


public class AdaBoostPLTestMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable> {
	private ArrayList<ArrayList<WeakClassifier>> classifiers;
	private double[] corWeights;
	
	//private static final Logger LOG = LoggerFactory.getLogger(AdaBoostingPLTestMapper.class);		
	
	protected void setup(Context context) throws IOException, InterruptedException{
		String src = context.getConfiguration().get("AdaBoost.ClassifiersFile"); //TODO
		
		classifiers = new ArrayList<ArrayList<WeakClassifier>>();
		
		FileSystem hdfs = FileSystem.get(context.getConfiguration());
		FSDataInputStream dis = hdfs.open(new Path(src));
		LineReader in = new LineReader(dis);
		Text line = new Text();
		while(in.readLine(line) > 0){
			ArrayList<WeakClassifier> ws = new ArrayList<WeakClassifier>();
			classifiers.add(ws);
			
			String[] items = line.toString().split(" ");
			for (int i = 0; i < items.length;i++) {
				WeakClassifier w = WeakClassifierHelper.newInstance();
				w.fromString(items[i]);
				ws.add(w);
			}
		}
		dis.close();
		in.close();
		
		corWeights = new double[classifiers.get(0).size()];
		for (int i = 0; i < classifiers.size(); i++) {
			for (int j = 0; j < classifiers.get(i).size(); j++) {
				corWeights[j] += classifiers.get(i).get(j).getCorWeight();
			}
		}
	}
	
	protected void map(LongWritable key, Text value, Context context) throws IOException ,InterruptedException {
		Instance inst = new Instance(value.toString());
		double H = 0;
		for (int i = 0; i < corWeights.length; i++) {
			H += corWeights[i] * merge(inst, i);
		}
		
		int hypoth = H >= 0.0 ? +1 : -1;
		if(hypoth == inst.getClassAttr()){
			context.getCounter("Error Rate", "Right number").increment(1);
		}
		else{
			context.getCounter("Error Rate", "Error number").increment(1);
		}
	}
	
	int merge(Instance inst, int round){
		int sum = 0;
		for (int i = 0; i < classifiers.size(); i++) {
			sum += classifiers.get(i).get(round).classifyInstance(inst);
		}
		
		if(sum == 0){
			return 1;
		}
		return sum > 0 ? +1 : -1;
	}
}
