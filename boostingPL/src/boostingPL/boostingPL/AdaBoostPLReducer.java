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

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import boostingPL.mr.io.WeakClassifierArrayWritable;


public class AdaBoostPLReducer extends Reducer<Text, WeakClassifierArrayWritable, 
	LongWritable, ArrayWritable>{
	
	protected void reduce(Text key, Iterable<WeakClassifierArrayWritable> value, Context context) throws IOException ,InterruptedException {
		if(key.toString().equals("H")){
			for(ArrayWritable item: value){
				context.write(null, item);
			}
		}
	}
}
