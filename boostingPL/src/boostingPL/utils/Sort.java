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

package boostingPL.utils;

import boostingPL.weakclassifier.WeakClassifier;

public class Sort {
	
	public static void sort(WeakClassifier[] classifiers, double[] orders) {
		quickSort(classifiers, orders, 0, classifiers.length-1);
	}
	
	private static void quickSort(WeakClassifier[] classifiers, double[] orders, int left, int right) {

		if(left < right) {
			int i = left;
			int j = right + 1;
			while(true) {
				while(i+1 < classifiers.length 
						&& orders[++i] < orders[left]);
				while(j-1 > -1
						&& orders[--j] > orders[left]);
				
				if (i >= j) {
					break;
				}
				swap(classifiers, orders, i, j);
			}
			swap(classifiers, orders, left, j);
			
			quickSort(classifiers, orders, left, j-1);
			quickSort(classifiers, orders, j+1, right);
		}
	}
	
	private static void swap(WeakClassifier[] classifiers,  double[] orders, int i, int j) {
		WeakClassifier tC;
		double t;
		
		tC = classifiers[i];
		classifiers[i] = classifiers[j];
		classifiers[j] = tC;
		
		t = orders[i];
		orders[i] = orders[j];
		orders[j] = t;
	}
}
