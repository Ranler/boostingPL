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

import boostingPL.core.Instances;
import boostingPL.weakclassifier.WeakClassifier;

/**
 * Logit Adaptive Boosting
 * 
 * @author Ranler Cao  findfunaax@gmail.com
 *
 */
public class LogitBoost {
	
	private Instances insts;
	
	private int numInterations;
	
	private WeakClassifier[] classifiers;
	

	public LogitBoost(Instances insts, int numInterations){
		this.insts = insts;
		this.numInterations = numInterations;
		this.classifiers = new WeakClassifier[numInterations];
	}
	
	public boolean run() {
		
		return true;
	}
}
