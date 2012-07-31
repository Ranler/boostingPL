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

package boostingPL.core;

public class Instance {
	private double[] attributes;
	private int classAttribute;

	public Instance(double[] attributes, int classAttribute) {
		this.attributes = attributes;
		this.classAttribute = classAttribute;
	}
	
	public Instance(String s) {
		String[] items = s.toString().split(" ");
		this.attributes = new double[items.length-1];
		for (int i = 0; i < items.length - 1 ; i++) {
			attributes[i] = Double.parseDouble(items[i]);
		}
		this.classAttribute = Integer.parseInt(items[items.length-1]);
	}
	
	public int attrNum() {
		return attributes.length;
	}
	
	public int getClassAttr() {
		return classAttribute;
	}
	
	public double[] getAttr() {
		return attributes;
	}
}
