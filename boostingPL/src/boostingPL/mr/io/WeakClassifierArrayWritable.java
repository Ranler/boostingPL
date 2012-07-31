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

package boostingPL.mr.io;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.Writable;

import boostingPL.weakclassifier.WeakClassifier;
import boostingPL.weakclassifier.WeakClassifierHelper;

public class WeakClassifierArrayWritable extends ArrayWritable{

	public WeakClassifierArrayWritable() {
		super(WeakClassifierHelper.getClassifierClass());
	}
	
	public WeakClassifierArrayWritable(WeakClassifier[] values) {
		super(WeakClassifierHelper.getClassifierClass());
		Writable[] vs = new Writable[values.length];
		for (int i = 0; i < vs.length; i++) {
			vs[i] = (Writable)values[i];
		}
		this.set(vs);
	}
	
	@Override
	public String toString() {
		Writable[] values = this.get();
		String s = "" + values[0];
		for (int i = 1; i < values.length; i++) {
			s += "|" + values[i].toString();
		}
		return s;
	}
}
