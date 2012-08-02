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

package boostingPL;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

import boostingPL.boostingPL.AdaBoostPL.AdaBoostPLDriver;
import boostingPL.boostingPL.LogitBoostPL.LogitBoostPLDriver;

/**
 * This Project is based on this paper:
 * 
 *   Indranil Palit and Chandan K. Reddy, "Scalable and Parallel Boosting
 *   with MapReduce", IEEE Transactions on Knowledge and Data Engineering
 *   (TKDE), 2012.
 *   
 * If you want to know the theory and demonstration of BoostingPL, this paper
 * provides references for further reading.
 * 
 * @author Ranler Cao  findfunaax@gmail.com
 */
public class BoostingPL {
	  
	private BoostingPL() {
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length > 0 && args[0].equals("adaboost")) {
			System.exit(ToolRunner.run(new Configuration(), new AdaBoostPLDriver(), args));
		}
		else if (args.length > 0 && args.equals("logitboost")) {
			System.exit(ToolRunner.run(new Configuration(), new LogitBoostPLDriver(), args));			
		}
		else {
			System.out.println("Usage: adaboost|logitboost");
		}
	}
}
