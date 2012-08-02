/*
 *   BoostPL - Scalable and Parallel Boosting with MapReduce 
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

package boostingPL.weakclassifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WeakClassifierHelper {
	/** default DecisionStump */
	private static Class<? extends WeakClassifier> nowClassifierClass = DecisionStump.class;
	
	private static final Logger log = LoggerFactory.getLogger(WeakClassifierHelper.class);
	
	public static void setClassifierClass(String name) {
		if (name.equals("DecisionStump")) {
			nowClassifierClass = DecisionStump.class;
		}
	}
	
	public static Class<? extends WeakClassifier> getClassifierClass() {
		return nowClassifierClass;
	}

	public static WeakClassifier newInstance() {
		try {
			return nowClassifierClass.newInstance();
		} catch (InstantiationException e) {
			log.warn("unable to new a weakclassifier instance", e);
		} catch (IllegalAccessException e) {
			log.warn("unable to new a weakclassifier instance", e);
		}
		return null;
	}
	


}
