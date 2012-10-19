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


package boostingPL.driver;

import org.apache.hadoop.util.ProgramDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class BoostingPLDriver {
	
	private static final Logger LOG = LoggerFactory.getLogger(BoostingPLDriver.class);	
	
	public static void main(String[] args) throws Throwable {
		ProgramDriver programDriver = new ProgramDriver();
		
		addClass(programDriver, "boostingPL.driver.AdaBoostPLDriver",
				"AdaBoostPL:Parallel boosting for binary classifier problem");
		addClass(programDriver, "boostingPL.driver.SAMMEPLDriver",
				"SAMMEPL:Parallel boosting for multiclass classifier problem");		
		
	    programDriver.driver(args);
	}
	
	private static void addClass(ProgramDriver driver, String classString, String descString) {
		try {
			Class<?> clazz = Class.forName(classString);
			driver.addClass(shortName(descString), clazz, desc(descString));
		} catch (ClassNotFoundException e) {
			LOG.warn("Unable to add class: {}", classString, e);
		} catch (Throwable t) {
			LOG.warn("Unable to add class: {}", classString, t);
		}
	}
	
	private static String shortName(String valueString) {
		return valueString.contains(":") ? valueString.substring(0, valueString.indexOf(':')).trim() : valueString;
	}

	private static String desc(String valueString) {
		return valueString.contains(":") ? valueString.substring(valueString.indexOf(':')+1).trim() : valueString;
	}
}
