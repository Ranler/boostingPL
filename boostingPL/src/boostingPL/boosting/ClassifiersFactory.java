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

import weka.classifiers.Classifier;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class ClassifiersFactory {

	public static Classifier newInstance(String classifierName) throws Exception {
		if (classifierName.equals("DecisionStump")) {
			return new DecisionStump();
		}
		if (classifierName.equals("C4.5")) {
			String arg = "weka.classifiers.trees.J48 -C 0.25 -M 2";
			J48 j48 = new J48();
			j48.setOptions(Utils.splitOptions(arg));
			return j48;
		}
		return null;
	}
}
